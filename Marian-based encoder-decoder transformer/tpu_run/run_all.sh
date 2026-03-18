#!/usr/bin/env bash
# One entrypoint: install deps then start TPU training.
# Re-run training without reinstall: SKIP_TPU_INSTALL=1 ./run_all.sh --data_csv ...
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${SKIP_TPU_INSTALL:-0}" != "1" ]]; then
  ./install_tpu_deps.sh
fi
export PJRT_DEVICE=TPU
exec python3 train_tpu.py "$@"
