import torch
from src.builder import BUILDER
from PIL import Image
from mmengine.config import Config
import argparse
from einops import rearrange
import json
import os


def generate_and_save(model, prompt, cfg_prompt, args, output_name):
    """给定 prompt 和 cfg_prompt，生成一张图并保存"""
    class_info = model.prepare_text_conditions(prompt, cfg_prompt)
    input_ids = class_info['input_ids']
    attention_mask = class_info['attention_mask']

    assert len(input_ids) == 2  # the last one is unconditional prompt

    if args.cfg == 1.0:
        input_ids = input_ids[:1]
        attention_mask = attention_mask[:1]
        bsz = 1
    else:
        bsz = 1
        input_ids = torch.cat([
            input_ids[:1].expand(bsz, -1),
            input_ids[1:].expand(bsz, -1),
        ])
        attention_mask = torch.cat([
            attention_mask[:1].expand(bsz, -1),
            attention_mask[1:].expand(bsz, -1),
        ])

    m = n = args.image_size // 16
    samples = model.sample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_iter=args.num_iter,
        cfg=args.cfg,
        cfg_schedule=args.cfg_schedule,
        temperature=args.temperature,
        progress=True,
        image_shape=(m, n)
    )

    # (c,h,w) -> (h,w,c)
    samples = rearrange(samples[0], 'c h w -> h w c')
    samples = torch.clamp(
        127.5 * samples + 128.0, 0, 255
    ).to("cpu", dtype=torch.uint8).numpy()
    Image.fromarray(samples).save(output_name)
    print(f"Saved: {output_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path.')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--json", type=str, default="prompts.json",
                        help="json file containing prompts list")
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument('--cfg_schedule', type=str, default='constant')
    parser.add_argument('--num_iter', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).eval().cuda()
    model = model.to(model.dtype)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint, strict=False)

    # consistency base description
    base_eval = "Consistency quality evaluates how smoothly the object integrates with its surroundings in terms of color, texture, and illumination."

    # 读取 JSON prompts
    with open(args.json, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # 遍历 prompts
    for idx, p in enumerate(prompts, 1):
        target = p["prompt"]

        prompts_and_cfgs = [
            (
                f"{base_eval} Generate an image with low consistency quality: {target}",
                f"{base_eval} Generate an image.",
                os.path.join(args.output_dir, f"output{idx}_low.jpg")
            ),
            (
                f"{base_eval} Generate an image with high consistency quality: {target}",
                f"{base_eval} Generate an image.",
                os.path.join(args.output_dir, f"output{idx}_high.jpg")
            ),
            (
                f"Generate an image: {target}",
                "Generate an image.",
                os.path.join(args.output_dir, f"output{idx}_normal.jpg")
            ),
        ]

        # 依次生成并保存
        for prompt_text, cfg_prompt, name in prompts_and_cfgs:
            generate_and_save(model, prompt_text, cfg_prompt, args, name)
