#!/usr/bin/env bash
# Run once on a Google Cloud TPU VM (SSH) before training.
# https://cloud.google.com/tpu/docs/run-calculation-pytorch
set -euo pipefail
cd "$(dirname "$0")"

# torch and torch_xla MUST be the same release (native ABI). Unpinned "torch"
# from PyPI + torch_xla gives _XLAC undefined symbol errors.
# Bump together when upgrading; check https://pypi.org/project/torch-xla/
# Override on the VM if needed:  TORCH_VER=2.8.0 ./install_tpu_deps.sh
TORCH_VER="${TORCH_VER:-2.9.0}"

sudo apt-get update
sudo apt-get install -y libopenblas-dev python3-venv

python3 -m pip install -q -U pip
python3 -m pip install -q -U numpy

# transformers pulls torch; avoid upgrading more than necessary before we pin.
python3 -m pip install -q -r requirements-tpu.txt --upgrade-strategy only-if-needed

python3 -m pip uninstall -y torch torchvision torchaudio torch_xla 2>/dev/null || true

python3 -m pip install -q --no-cache-dir \
  "torch==${TORCH_VER}" "torch_xla[tpu]==${TORCH_VER}" \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://storage.googleapis.com/libtpu-releases/index.html

echo "Verifying TPU devices..."
export PJRT_DEVICE=TPU
python3 -c "import torch_xla.core.xla_model as xm; print('TPU devices:', xm.get_xla_supported_devices('TPU'))"

echo "Done. If this is the first install, restart your shell or reboot before long training runs."
