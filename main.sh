#!/bin/bash
# main.sh
# Usage: bash main.sh <MODEL_ABBR>

MODEL_ABBR=$1    # 模型缩写，例如 umm
STORAGE_PATH=./checkpoints/${MODEL_ABBR}
mkdir -p $STORAGE_PATH

# ==================== Iteration 1 ====================
echo "==== Iteration 1 ===="
bash scripts/train1.sh round1 q
bash scripts/train2.sh round1 s \
    --q_ckpt ${STORAGE_PATH}/round1/q/global_step_5/actor/huggingface

# ==================== Iteration 2~5 ====================
for i in {2..5}; do
    prev=$((i-1))
    ROUND_ID="round${i}"
    PREV_ROUND="round${prev}"

    echo "==== Iteration $i ===="

    # train1: 问题模型
    bash scripts/train1.sh \
        $ROUND_ID q \
        --prev_q ${STORAGE_PATH}/${PREV_ROUND}/q/global_step_5/actor/huggingface \
        --solver_ckpt ${STORAGE_PATH}/${PREV_ROUND}/s/global_step_15/actor/huggingface

    # train2: 解答模型
    bash scripts/train2.sh \
        $ROUND_ID s \
        --q_ckpt ${STORAGE_PATH}/${ROUND_ID}/q/global_step_5/actor/huggingface \
        --solver_ckpt ${STORAGE_PATH}/${PREV_ROUND}/s/global_step_15/actor/huggingface
done

# ==================== Final Evaluation ====================
bash evaluation/evaluate.bash
