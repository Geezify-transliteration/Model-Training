#!/usr/bin/env bash
# Re-pin torch + torch_xla inside the CURRENT python/pip (e.g. after venv activate).
# Use when you see: ImportError ... _XLAC ... undefined symbol ... torch::Library
set -euo pipefail
cd "$(dirname "$0")"

TORCH_VER="${TORCH_VER:-2.9.0}"

python3 -m pip uninstall -y torch torchvision torchaudio torch_xla 2>/dev/null || true
python3 -m pip install --no-cache-dir \
  "torch==${TORCH_VER}" "torch_xla[tpu]==${TORCH_VER}" \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://storage.googleapis.com/libtpu-releases/index.html

python3 -c "import torch; import torch_xla; print('torch', torch.__version__, 'OK')"
