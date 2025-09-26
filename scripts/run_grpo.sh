export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=3,4,5

RUN_NAME="Qwen2.5-VL-3B-GRPO-REC"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="3" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    open_r1/grpo_rec.py \
    --deepspeed open_r1/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /hd2/tangzhenchen/model/harmon/harmon/ \
    --dataset_name open_r1/rec.yaml \
    --image_root "/hd2/wangzichuan/CLIP-AGIQA/AGIQA-3K" \
    --max_prompt_length 1024 \
    --num_generations 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 True \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true