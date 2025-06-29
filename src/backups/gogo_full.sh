#!/bin/bash

set -e  # 出錯即停止


GPUS_PER_NODE=2
SUBSET_RATIO=1
PYTHON_SCRIPT="run_rnnlogic.py"

CLUSTER_SIZES=(
  3
  # 5
  # 6
  # 7
  # 8
)

DATASET="semmeddb" 
# DATASET="wn18rr" 

CONFIG_LIST=(
  # "../config/sm.yaml"
  # "../config/md.yaml"
  "../config/full.yaml"
)

LOSS_MODES=(
  # "ori"
  # "fixed"
  # "warmup"
  # "schedule"
  "adaptive"
)

RELATION_CLUSTER_METHODS=(
  # "naive"
  "matrix"
)

PREDICTOR_WEIGHTED_LOSS_MODES=(
  # "ori"
  "triple_count"
  "triple_count_sqrt"
  # "triple_count_log"
)

IS_WRNNLOGIC=0  # 是否使用 WRNNLogic 模型


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

  for LOSS_MODE in "${LOSS_MODES[@]}"; do
    for RELATION_CLUSTER_METHOD in "${RELATION_CLUSTER_METHODS[@]}"; do
      for CLUSTER_SIZE in "${CLUSTER_SIZES[@]}"; do
        for PREDICTOR_WEIGHTED_LOSS_MODE in "${PREDICTOR_WEIGHTED_LOSS_MODES[@]}"; do
          STAMP=$(date +"%Y%m%d_%H%M%S")
          echo "🔧 Time stamp: $STAMP, Processing config: $CONFIG_FILENAME with LOSS_MODE: $LOSS_MODE, RELATION_CLUSTER_METHOD: $RELATION_CLUSTER_METHOD, CLUSTER_SIZE: $CLUSTER_SIZE, PREDICTOR_WEIGHTED_LOSS_MODE: $PREDICTOR_WEIGHTED_LOSS_MODE"
          CONFIG_PATH="../config/copies_in_process/${LOSS_MODE}_${CONFIG_FILENAME}.yaml"
          cp "$CONFIG_ORIGINAL" "$CONFIG_PATH"

          if [[ "$RUN_MODE" == "torchrun" ]]; then
            echo "🔧 Updating config for torchrun..."
            sed -i '/^[[:space:]]*gpus:/c\  gpus: [0, 1]' "$CONFIG_PATH"
          else
            echo "🔧 Updating config for normal python run..."
            sed -i '/^[[:space:]]*gpus:/c\  gpus: [0]' "$CONFIG_PATH"
          fi

          # sed -i "s|^save_path: .*|save_path: results/${DATASET}/multitask/${RELATION_CLUSTER_METHOD}_encode_relation/task1_next_clus/random_init/cluster_${CLUSTER_SIZE}/${LOSS_MODE}_with_count|" "$CONFIG_PATH"
          # sed -i "s|^save_path: .*|save_path: results/${DATASET}/multitask/${RELATION_CLUSTER_METHOD}_encode_relation/task1_next_clus/random_init/cluster_${CLUSTER_SIZE}/test_all/${LOSS_MODE}_with_count|" "$CONFIG_PATH"
          sed -i "s|^save_path: .*|save_path: results/${DATASET}/multitask/${RELATION_CLUSTER_METHOD}_encode_relation/task1_next_clus/random_init/cluster_${CLUSTER_SIZE}/0618_all_filtered/${LOSS_MODE}/${PREDICTOR_WEIGHTED_LOSS_MODE}_${STAMP}|" "$CONFIG_PATH"
          # sed -i "s|^save_path: .*|save_path: results/${DATASET}/wRNNLogic_${STAMP}|" "$CONFIG_PATH"
          # sed -i "s|^save_path: .*|save_path: results/test|" "$CONFIG_PATH"

          sed -i "s|data_path: .*|data_path: ../data/${DATASET}|" "$CONFIG_PATH"

          if [[ "$IS_WRNNLOGIC" -eq 1 ]]; then
            sed -i "s|rule_file: .*|rule_file: ../data/${DATASET}/weitsu_rules.txt|" "$CONFIG_PATH"
          else
            sed -i "s|rule_file: .*|rule_file: ../data/${DATASET}/mined_rules.txt|" "$CONFIG_PATH"
          fi
          
          sed -i "s|cluster_size: .*|cluster_size: ${CLUSTER_SIZE}|" "$CONFIG_PATH"
          sed -i "s|relation_cluster_file: .*|relation_cluster_file: ../data/${DATASET}/relation_cluster/${RELATION_CLUSTER_METHOD}_${CLUSTER_SIZE}.dict|" "$CONFIG_PATH"

          EXP_NAME="${DATASET}_${LOSS_MODE}_${SIZE}"
          echo "▶ Running: $EXP_NAME"

          LOG_FILE="logs/${EXP_NAME}.log"

          if [[ "$RUN_MODE" == "torchrun" ]]; then
            echo "🚀 Running with torchrun..."
            if ! torchrun \
              --nproc-per-node=$GPUS_PER_NODE $PYTHON_SCRIPT \
              --config $CONFIG_PATH \
              --mode $LOSS_MODE \
              --subset_ratio $SUBSET_RATIO \
              --predictor_weighted_loss_mode $PREDICTOR_WEIGHTED_LOSS_MODE \
              > "$LOG_FILE" 2>&1
            then
              echo "❌ Error detected in $EXP_NAME. See log: $LOG_FILE"
              exit 1
            fi

          else
            echo "🚀 Running with python..."
            if ! CUDA_LAUNCH_BLOCKING=1 python $PYTHON_SCRIPT \
              --config $CONFIG_PATH \
              --mode $LOSS_MODE \
              --subset_ratio $SUBSET_RATIO \
              --predictor_weighted_loss_mode $PREDICTOR_WEIGHTED_LOSS_MODE \
              > "$LOG_FILE" 2>&1
            then
              echo "❌ Error detected in $EXP_NAME. See log: $LOG_FILE"
              exit 1
            fi
          fi


          echo "✅ Finished: $EXP_NAME"
        done
      done
    done
  done
done

echo "🎉 All experiments completed."
