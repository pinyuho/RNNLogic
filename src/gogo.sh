#!/bin/bash

GPUS_PER_NODE=2
PYTHON_SCRIPT="run_rnnlogic.py"

DATASET="semmed" 

# 三種 config 主檔（small / mid / full）
CONFIG_LIST=(
  "../config/sm.yaml"
  # "../config/md.yaml"
  # "../config/full.yaml"
)

MODES=(
  "ori"
  # "fixed"
  # "warmup"
  # "schedule"
  # "adaptive"
)

mkdir -p logs

for CONFIG_ORIGINAL in "${CONFIG_LIST[@]}"; do
  # 取出 config 檔名（不含路徑與 .yaml）
  CONFIG_FILENAME=$(basename "$CONFIG_ORIGINAL" .yaml)
  SIZE=$(echo "$CONFIG_FILENAME" | grep -oE 'sm|md|full')

  for MODE in "${MODES[@]}"; do
    CONFIG_NAME="${MODE}_${CONFIG_FILENAME}.yaml"
    CONFIG_PATH="../config/copies_in_process/${CONFIG_NAME}"

    # ⏩ 複製並替換 save_path 為帶 mode + size 的版本
    cp "$CONFIG_ORIGINAL" "$CONFIG_PATH"
    # sed -i "s|^save_path: .*|save_path: results/${DATASET}/multitask/${MODE}_${SIZE}|" "$CONFIG_PATH"
    sed -i "s|data_path: .*|data_path: ../data/${DATASET}|" "$CONFIG_PATH"
    sed -i "s|rule_file: .*|rule_file: ../data/${DATASET}/mined_rules.txt|" "$CONFIG_PATH"

    EXP_NAME="${DATASET}_${MODE}_${SIZE}"
    echo "▶ Running: $EXP_NAME"

    # python $PYTHON_SCRIPT --config $CONFIG_PATH --mode $MODE \
    #   > logs/${EXP_NAME}.log 2>&1

    torchrun \
      --nproc-per-node=$GPUS_PER_NODE $PYTHON_SCRIPT \
      --config $CONFIG_PATH \
      --mode $MODE \
      > logs/${EXP_NAME}.log 2>&1

    echo "✅ Finished: $EXP_NAME"
  done
done

echo "🎉 All experiments completed."
