import numpy as np
import torch
from PIL import Image
from mmengine.config import Config
from src.builder import BUILDER
from einops import rearrange
import argparse


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def preprocess_image(path, image_size, model):
    image = Image.open(path).convert('RGB')
    image = expand2square(image, (127, 127, 127))
    image = image.resize(size=(image_size, image_size))
    image = torch.from_numpy(np.array(image)).to(dtype=model.dtype, device=model.device)
    image = rearrange(image, 'h w c -> c h w')[None]
    image = 2 * (image / 255) - 1
    return image


def build_inputs(prompt, image, model, image_token_idx, image_size):
    """build prompt with single image"""
    prompt_text = model.prompt_template['INSTRUCTION'].format(
        input="<image>\n" + prompt
    )

    image_length = (image_size // 16) ** 2 + 64
    prompt_text = prompt_text.replace('<image>', '<image>' * image_length)

    input_ids = model.tokenizer.encode(
        prompt_text, add_special_tokens=True, return_tensors='pt'
    ).cuda()

    with torch.no_grad():
        _, z_enc = model.extract_visual_feature(model.encode(image))

    inputs_embeds = z_enc.new_zeros(
        *input_ids.shape, model.llm.config.hidden_size
    )

    image_positions = (input_ids == image_token_idx).nonzero(as_tuple=True)[1]
    inputs_embeds[0, image_positions] = z_enc.flatten(0, 1)

    # text tokens
    inputs_embeds[input_ids != image_token_idx] = model.llm.get_input_embeddings()(
        input_ids[input_ids != image_token_idx]
    )

    return input_ids, inputs_embeds


def chat_round(model, state, prompt, image, image_token_idx, image_size):
    """One round of QA with state"""
    input_ids, inputs_embeds = build_inputs(prompt, image, model, image_token_idx, image_size)

    if state.get("past_ids") is not None:
        # 拼接历史 input_ids
        input_ids = torch.cat([state["past_ids"], input_ids], dim=1)
        inputs_embeds = torch.cat([state["past_inputs_embeds"], inputs_embeds], dim=1)

    print(input_ids.size())
    print(inputs_embeds.size())
    with torch.no_grad():
        output = model.llm.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=state.get("past_key_values", None),
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
            do_sample=False,
            max_new_tokens=512,
            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
        )

    # 更新 state
    state["past_key_values"] = output.past_key_values
    state["past_ids"] = output.sequences[:, :-1]
    state["past_inputs_embeds"] = output.hidden_states[-1][0]

    ans = model.tokenizer.decode(output.sequences[0, input_ids.size(1):], skip_special_tokens=True)
    return ans, state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path.')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image1", type=str, default="data/view1.jpg")
    parser.add_argument("--image2", type=str, default="data/view2.jpg")
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).eval().cuda()
    model = model.to(model.dtype)

    if args.checkpoint is not None:
        print(f"Load checkpoint: {args.checkpoint}", flush=True)
        checkpoint = torch.load(args.checkpoint)
        _ = model.load_state_dict(checkpoint, strict=False)

    # register <image>
    special_tokens_dict = {'additional_special_tokens': ["<image>", ]}
    num_added_toks = model.tokenizer.add_special_tokens(special_tokens_dict)
    assert num_added_toks == 1
    image_token_idx = model.tokenizer.encode("<image>", add_special_tokens=False)[-1]

    # preprocess images
    image1 = preprocess_image(args.image1, args.image_size, model)
    image2 = preprocess_image(args.image2, args.image_size, model)

    # state 初始化
    state = {}

    # round 1
    q1 = "Consistency quality refers to the overall visual coherence of an image, including the seamless integration of all regions in terms of background, lighting, color, and texture. A high consistency quality means the elements do not appear mismatched or disconnected, but instead form a unified and natural whole. Describe the consistency quality of image1."
    ans1, state = chat_round(model, state, q1, image1, image_token_idx, args.image_size)
    print(f"\nUser: {q1}\nAssistant: {ans1}")

    # round 2
    q2 = "Describe the consistency quality of image2. Compare it with the previous analysis of image1, assuming both images show the same objects or scene. Answer the question: Is image2 better than image1 in consistency quality? Provide a one-sentence reason followed by 'Yes' or 'No'."
    ans2, state = chat_round(model, state, q2, image2, image_token_idx, args.image_size)
    print(f"\nUser: {q2}\nAssistant: {ans2}")
