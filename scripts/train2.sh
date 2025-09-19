#!/bin/bash
# train.sh <ROUND_ID> <ROLE_NAME>

ROUND_ID=$1
ROLE_NAME=$2
shift 2  # 其他参数透传给 train.py

export ROUND_ID
export ROLE_NAME

MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
GPUS_PER_NODE=1
NODE_RANK=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONFIG_FILE="configs/examples/evo.py"

export LAUNCHER="torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    "

export CMD="scripts/train.py \
$CONFIG_FILE \
--launcher pytorch \
--deepspeed deepspeed_zero2"

echo $LAUNCHER
echo $CMD

bash -c "$LAUNCHER $CMD"

sleep 60s