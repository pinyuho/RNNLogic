#!/bin/bash

set -e  # 出錯即停止
mkdir -p logs

RUN_MODE="$1"  # 例如： bash gogo.sh torchrun 或 bash gogo.sh normal
IS_SOFT_LABEL=False
IS_SCHEDULED_SAMPLING=False
IS_TEST_MODE=False # FIXME:

TYPE_OR_GROUP="type"

USE_GPU=1

GPUS_PER_NODE=2
SUBSET_RATIO=1
PYTHON_SCRIPT="run_rnnlogic.py"

DATASET="semmeddb" 

CONFIG_ORIGINAL="../config/full.yaml" # FIXME:
CONFIG_FILENAME=$(basename "$CONFIG_ORIGINAL" .yaml)

# MULTITASK_LOSS_MODE="adaptive"  # fixed, warmup, schedule
MULTITASK_LOSS_MODE="adaptive"
PREDICTOR_WEIGHTED_LOSS_MODE="ori"  # triple_count, triple_count_sqrt, triple_count_log
RELATION_CLUSTER_METHOD="matrix" # naive, matrix

CLUSTER_SIZES=(
  3
  # 5
  # 6
  # 7
  # 8
)

MODEL=(
  # "ori"
  "multitask_hard_sharing"
  # "multitask_mmoe"
)

if [[ "$RUN_MODE" != "normal" && "$RUN_MODE" != "torchrun" ]]; then
  echo "Usage: $0 [normal|torchrun]"
  exit 1
fi



for CLUSTER_SIZE in "${CLUSTER_SIZES[@]}"; do
  STAMP=$(date +"%Y%m%d_%H%M%S")

  echo "============================================================="
  echo " 🚀 Starting run with the following configuration:"
  echo "-------------------------------------------------------------"
  echo " 🔹 Config                  : $CONFIG_FILENAME.yaml"
  echo " 🔹 Model                   : $MODEL"
  echo " 🔹 Is Test Mode            : $IS_TEST_MODE"
  echo " 🔹 Loss Mode               : $MULTITASK_LOSS_MODE"
  echo " 🔹 Predictor Loss          : $PREDICTOR_WEIGHTED_LOSS_MODE"
  echo " 🔹 Relation Clust          : $RELATION_CLUSTER_METHOD (cluster size: $CLUSTER_SIZE)"
  echo " 🔹 Is Soft Label           : $IS_SOFT_LABEL"
  echo " 🔹 Is Scheduled sampling   : $IS_SCHEDULED_SAMPLING"
  echo " 🔹 Dataset                 : $DATASET"
  echo " 🔹 Semantic Type / Group   : $TYPE_OR_GROUP"
  echo " 🔹 Subset Ratio            : $SUBSET_RATIO"
  echo " 🔹 GPUs Per Node           : $GPUS_PER_NODE"
  echo " 🔹 Run Mode                : $RUN_MODE"
  echo "============================================================="

  CONFIG_PATH="../config/copies_in_process/${MULTITASK_LOSS_MODE}_${CONFIG_FILENAME}.yaml"
  cp "$CONFIG_ORIGINAL" "$CONFIG_PATH"

  if [[ "$RUN_MODE" == "torchrun" ]]; then
    echo "🔧 Updating config for torchrun..."
    sed -i '/^[[:space:]]*gpus:/c\  gpus: [0, 1]' "$CONFIG_PATH"
  else
    echo "🔧 Updating config for normal python run..."
    sed -i "/^[[:space:]]*gpus:/c\  gpus: [${USE_GPU}]" "$CONFIG_PATH"
  fi


  EXP_NAME="${MODEL}_${MULTITASK_LOSS_MODE}_${RELATION_CLUSTER_METHOD}_${CLUSTER_SIZE}_${TYPE_OR_GROUP}"
  echo "▶ Running: ${STAMP}.log"
  LOG_FILE="logs/${STAMP}.log"

  sed -i "s|^save_path: .*|save_path: results/${DATASET}/multitask_mmoe/ent_all_types/Aux2_${EXP_NAME}_${STAMP}|" "$CONFIG_PATH"
  # sed -i "s|^save_path: .*|save_path: results/${DATASET}/sm_test/${STAMP}|" "$CONFIG_PATH" # FIXME:

  sed -i "s|data_path: .*|data_path: ../data/${DATASET}|" "$CONFIG_PATH"
  sed -i "s|rule_file: .*|rule_file: ../data/${DATASET}/mined_rules.txt|" "$CONFIG_PATH"
  sed -i "s|cluster_size: .*|cluster_size: ${CLUSTER_SIZE}|" "$CONFIG_PATH"
  # sed -i "s|relation_cluster_file: .*|relation_cluster_file: ../data/${DATASET}/relation_cluster/${RELATION_CLUSTER_METHOD}_${CLUSTER_SIZE}.dict|" "$CONFIG_PATH"
  if [ "$TYPE_OR_GROUP" == "group" ]; then
    sed -i "s|relation_cluster_file: .*|relation_cluster_file: ../data/${DATASET}/relation_cluster/sg_${RELATION_CLUSTER_METHOD}_${CLUSTER_SIZE}.dict|" "$CONFIG_PATH"
  else
    sed -i "s|relation_cluster_file: .*|relation_cluster_file: ../data/${DATASET}/relation_cluster/${RELATION_CLUSTER_METHOD}_${CLUSTER_SIZE}.dict|" "$CONFIG_PATH"
  fi

  if [[ "$RUN_MODE" == "torchrun" ]]; then
    echo "🚀 Running with torchrun..."
    if ! torchrun \
      --nproc-per-node=$GPUS_PER_NODE $PYTHON_SCRIPT \
      --config $CONFIG_PATH \
      --subset_ratio $SUBSET_RATIO \
      --model $MODEL \
      --multitask_loss_mode $MULTITASK_LOSS_MODE \
      --predictor_weighted_loss_mode $PREDICTOR_WEIGHTED_LOSS_MODE \
      --is_soft_label $IS_SOFT_LABEL \
      --is_scheduled_sampling $IS_SCHEDULED_SAMPLING \
      --type_or_group $TYPE_OR_GROUP \
      --is_test_mode $IS_TEST_MODE \
      > "$LOG_FILE" 2>&1
    then
      echo "❌ Error detected in $EXP_NAME. See log: $LOG_FILE"
      exit 1
    fi

  else
    echo "🚀 Running with python..."
    if ! CUDA_LAUNCH_BLOCKING=1 python $PYTHON_SCRIPT \
      --config $CONFIG_PATH \
      --subset_ratio $SUBSET_RATIO \
      --model $MODEL \
      --multitask_loss_mode $MULTITASK_LOSS_MODE \
      --predictor_weighted_loss_mode $PREDICTOR_WEIGHTED_LOSS_MODE \
      --is_soft_label $IS_SOFT_LABEL \
      --is_scheduled_sampling $IS_SCHEDULED_SAMPLING \
      --type_or_group $TYPE_OR_GROUP \
      --is_test_mode $IS_TEST_MODE \
      > "$LOG_FILE" 2>&1
    then
      echo "❌ Error detected in $EXP_NAME. See log: $LOG_FILE"
      exit 1
    fi
  fi
  echo "✅ Finished: $EXP_NAME"
done

echo "🎉 All experiments completed."
