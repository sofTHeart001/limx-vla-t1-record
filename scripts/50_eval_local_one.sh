#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

set_track_vars "${1:-}"
CKPT_DIR="${2:-$(default_ckpt_dir)}"

require_robotwin
require_path "${ROOT_DIR}/starter/eval_local.py"
require_path "${CKPT_DIR}"

extra_args=()
if [[ "${TRACK}" == "T3" ]]; then
  if [[ ! -f "${ACT_DIR}/deploy_t3.yml" && -f "${ROOT_DIR}/configs/deploy_t3.yml" ]]; then
    cp "${ROOT_DIR}/configs/deploy_t3.yml" "${ACT_DIR}/deploy_t3.yml"
  fi
  if [[ -f "${ACT_DIR}/deploy_t3.yml" ]]; then
    extra_args+=(--deploy-config policy/ACT/deploy_t3.yml)
  fi
fi

info "本地自评 ${TRACK}: ckpt_dir=${CKPT_DIR}"
(cd "${ROOT_DIR}" && python starter/eval_local.py --track "${TRACK}" --ckpt-dir "${CKPT_DIR}" "${extra_args[@]}")
