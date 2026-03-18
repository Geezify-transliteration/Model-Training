#!/usr/bin/env bash
# Run once on a Google Cloud TPU VM (SSH) before training.
# https://cloud.google.com/tpu/docs/run-calculation-pytorch
set -euo pipefail
cd "$(dirname "$0")"

sudo apt-get update
sudo apt-get install -y libopenblas-dev

python3 -m pip install -q -U pip
python3 -m pip install -q -U numpy
python3 -m pip install -U torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
python3 -m pip install -r requirements-tpu.txt

echo "Verifying TPU devices..."
export PJRT_DEVICE=TPU
python3 -c "import torch_xla.core.xla_model as xm; print('TPU devices:', xm.get_xla_supported_devices('TPU'))"

echo "Done. If this is the first install, restart your shell or reboot before long training runs."
