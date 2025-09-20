#!/bin/bash
# main.sh
# Usage: bash main.sh <MODEL_ABBR>

STORAGE_PATH=./checkpoints
mkdir -p $STORAGE_PATH

# ==================== Iteration 1 ====================
echo "==== Iteration 1 ===="
bash scripts/train1.sh ${STORAGE_PATH}/q/round1 /hd2/tangzhenchen/model/harmon/Harmon_1.5b_ReAlign.pth
bash scripts/train2.sh ${STORAGE_PATH}/s/round1 ${STORAGE_PATH}/q/round1/harmon_1.5b.pth

# ==================== Iteration 2~5 ====================
for i in {2..5}; do
    prev=$((i-1))
    ROUND_ID="round${i}"
    PREV_ROUND="round${prev}"

    echo "==== Iteration $i ===="

    # train1: 问题模型
    bash scripts/train1.sh \
        ${STORAGE_PATH}/q/$ROUND_ID  \
         ${STORAGE_PATH}/s/$PREV_ROUND/harmon_1.5b.pth

    # train2: 解答模型
    bash scripts/train2.sh \
        ${STORAGE_PATH}/s/$ROUND_ID \
        ${STORAGE_PATH}/q/$ROUND_ID/harmon_1.5b.pth
done

# ==================== Final Evaluation ====================
bash evaluation/evaluate.bash
