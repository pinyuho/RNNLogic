#!/bin/bash

set -e  # 出錯即停止

GPUS_PER_NODE=2
CLUSTER_SIZE=4 # TODO: 改這 -> 2
SUBSET_RATIO=0.1
PYTHON_SCRIPT="run_rnnlogic.py"

DATASET="semmeddb" 
# DATASET="wn18rr" 

CONFIG_LIST=(
  # "../config/sm.yaml"
  # "../config/md.yaml"
  "../config/full.yaml"
)

MODES=(
  # "ori"

  # "fixed"
  # "warmup"
  # "schedule"
  "adaptive"
)

mkdir -p logs

RUN_MODE="$1"  # 例如： bash gogo.sh torchrun 或 bash gogo.sh normal

if [[ "$RUN_MODE" != "normal" && "$RUN_MODE" != "torchrun" ]]; then
  echo "Usage: $0 [normal|torchrun]"
  exit 1
fi

for CONFIG_ORIGINAL in "${CONFIG_LIST[@]}"; do
  # 取出 config 檔名（不含路徑與 .yaml）
  CONFIG_FILENAME=$(basename "$CONFIG_ORIGINAL" .yaml)
  SIZE=$(echo "$CONFIG_FILENAME" | grep -oE 'sm|md|full')

  for MODE in "${MODES[@]}"; do
    CONFIG_PATH="../config/copies_in_process/${MODE}_${CONFIG_FILENAME}.yaml"
    cp "$CONFIG_ORIGINAL" "$CONFIG_PATH"

    if [[ "$RUN_MODE" == "torchrun" ]]; then
      echo "🔧 Updating config for torchrun..."
      sed -i '/^[[:space:]]*gpus:/c\  gpus: [0, 1]' "$CONFIG_PATH"
    else
      echo "🔧 Updating config for normal python run..."
      sed -i '/^[[:space:]]*gpus:/c\  gpus: [0]' "$CONFIG_PATH"
    fi

    sed -i "s|^save_path: .*|save_path: results/${DATASET}/sm_test/random_init/matrix_svd_cluster_${CLUSTER_SIZE}/${MODE}|" "$CONFIG_PATH" # TODO: 改這 -> bidirectional_...

    sed -i "s|data_path: .*|data_path: ../data/${DATASET}|" "$CONFIG_PATH"
    sed -i "s|rule_file: .*|rule_file: ../data/${DATASET}/mined_rules.txt|" "$CONFIG_PATH"
    sed -i "s|cluster_size: .*|cluster_size: ${CLUSTER_SIZE}|" "$CONFIG_PATH"
    sed -i "s|relation_cluster_file: .*|relation_cluster_file: ../data/${DATASET}/relation_cluster/matrix_svd_${CLUSTER_SIZE}.dict|" "$CONFIG_PATH" # TODO: 改這 -> bidirectional_...

    # grep "gpus:" "$CONFIG_PATH"

    EXP_NAME="${DATASET}_${MODE}_${SIZE}"
    echo "▶ Running: $EXP_NAME"

    LOG_FILE="logs/${EXP_NAME}.log"

    if [[ "$RUN_MODE" == "torchrun" ]]; then
      echo "🚀 Running with torchrun..."
      if ! torchrun \
        --nproc-per-node=$GPUS_PER_NODE $PYTHON_SCRIPT \
        --config $CONFIG_PATH \
        --mode $MODE \
        --subset_ratio $SUBSET_RATIO \
        > "$LOG_FILE" 2>&1
      then
        echo "❌ Error detected in $EXP_NAME. See log: $LOG_FILE"
        exit 1
      fi

    else
      echo "🚀 Running with python..."
      if ! CUDA_LAUNCH_BLOCKING=1 python $PYTHON_SCRIPT \
        --config $CONFIG_PATH \
        --mode $MODE \
        --subset_ratio $SUBSET_RATIO \
        > "$LOG_FILE" 2>&1
      then
        echo "❌ Error detected in $EXP_NAME. See log: $LOG_FILE"
        exit 1
      fi
    fi


    echo "✅ Finished: $EXP_NAME"
  done
done

echo "🎉 All experiments completed."
