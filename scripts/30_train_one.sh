#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

set_track_vars "${1:-}"
SEED="${2:-0}"
GPU_ID="${3:-0}"

require_act
require_path "${ACT_DIR}/train.sh"

if [[ "${TRACK}" == "T3" ]]; then
  export ACT_BATCH_SIZE="${ACT_BATCH_SIZE:-24}"
  export ACT_CHUNK_SIZE="${ACT_CHUNK_SIZE:-100}"
  export ACT_LR="${ACT_LR:-2e-5}"
  export ACT_LR_BACKBONE="${ACT_LR_BACKBONE:-3e-6}"
  export ACT_WEIGHT_DECAY="${ACT_WEIGHT_DECAY:-1e-4}"
  export ACT_WARMUP_STEPS="${ACT_WARMUP_STEPS:-500}"
  export ACT_GRAD_CLIP="${ACT_GRAD_CLIP:-0.1}"
  export ACT_VAL_RATIO="${ACT_VAL_RATIO:-0.1}"
  export ACT_KL_WEIGHT="${ACT_KL_WEIGHT:-10}"
  export ACT_NUM_EPOCHS="${ACT_NUM_EPOCHS:-4000}"
  export ACT_SAVE_FREQ="${ACT_SAVE_FREQ:-1000}"
  export ACT_AUG="${ACT_AUG:-1}"
fi

info "训练 ${TRACK}: ${TASK_NAME} ${TASK_CONFIG} ${EXPERT_DATA_NUM} seed=${SEED} GPU=${GPU_ID} batch=${ACT_BATCH_SIZE:-8} chunk=${ACT_CHUNK_SIZE:-50} lr=${ACT_LR:-1e-5} backbone_lr=${ACT_LR_BACKBONE:-1e-5} wd=${ACT_WEIGHT_DECAY:-1e-4} warmup=${ACT_WARMUP_STEPS:-0} grad_clip=${ACT_GRAD_CLIP:-0} val_ratio=${ACT_VAL_RATIO:-0.2} epochs=${ACT_NUM_EPOCHS:-6000} aug=${ACT_AUG:-0}"
(cd "${ACT_DIR}" && bash train.sh "${TASK_NAME}" "${TASK_CONFIG}" "${EXPERT_DATA_NUM}" "${SEED}" "${GPU_ID}")
