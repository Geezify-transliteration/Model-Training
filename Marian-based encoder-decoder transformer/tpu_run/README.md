# TPU training (`tpu_run`)

Train the mT5 transliteration model on a **Google Cloud TPU VM** with PyTorch/XLA (no Colab, no `google.colab`).

Official PyTorch-on-TPU reference: [Run a calculation on a Cloud TPU VM using PyTorch](https://cloud.google.com/tpu/docs/run-calculation-pytorch).

---

## What you need

- A **TPU VM** you can SSH into (`gcloud compute tpus tpu-vm ssh ...`).
- Your dataset CSV on the VM (see below).
- **`sudo`** on the VM for first-time `apt` packages (`install_tpu_deps.sh`).

---

## Quick start

```bash
cd tpu_run
chmod +x install_tpu_deps.sh run_all.sh repair_tpu_torch_abi.sh
./run_all.sh --data_csv ./data/merged_normalized.csv
```

Train again **without** reinstalling wheels:

```bash
SKIP_TPU_INSTALL=1 ./run_all.sh --data_csv ./data/merged_normalized.csv
```

---

## Dataset

CSV path is passed with `--data_csv`. Required columns:

- `romanized_text`
- `amharic_text`

Example: copy data to the VM (from your laptop):

```bash
mkdir -p data
scp merged_normalized.csv USER@TPU_VM:~/path/to/tpu_run/data/
```

---

## Scripts

| File | Role |
|------|------|
| **`install_tpu_deps.sh`** | One-time (or after dependency changes): `apt` packages + `pip` installs PyTorch/XLA and Python deps. Pins matching **`torch`** / **`torch_xla`** versions. |
| **`repair_tpu_torch_abi.sh`** | Fast fix when `_XLAC` / `undefined symbol` / `torch::Library` appears: reinstalls the pinned pair using Google’s find-links (run inside your **venv** after `activate`). |
| **`run_all.sh`** | Runs `install_tpu_deps.sh` unless `SKIP_TPU_INSTALL=1`, sets `PJRT_DEVICE=TPU`, then `python3 train_tpu.py` with your args. |
| **`train_tpu.py`** | Training entrypoint; accepts `--data_csv`, `--output_dir`, `--gcs_bucket`, etc. |

---

## What `install_tpu_deps.sh` does

1. **`apt-get install`** `libopenblas-dev` and **`python3-venv`** (needed if you use `python3 -m venv` on Debian/Ubuntu images).
2. **`pip`**: upgrade `pip`, install `numpy`.
3. **`pip install -r requirements-tpu.txt`** with **`--upgrade-strategy only-if-needed`** (still pulls `torch` as a dependency of `transformers`).
4. **`pip uninstall`** `torch`, `torchvision`, `torchaudio`, `torch_xla` so no stray build remains.
5. **`pip install torch==X torch_xla[tpu]==X`** with **two** Google find-links: [`libtpu-wheels`](https://storage.googleapis.com/libtpu-wheels/index.html) and [`libtpu-releases`](https://storage.googleapis.com/libtpu-releases/index.html).

The script pins **`TORCH_VER` (default `2.9.0`)** so `torch` and `torch_xla` always share one ABI. Unpinned installs often produce:

`ImportError: ... _XLAC ... undefined symbol ... torch::Library`

because **`pip` may upgrade `torch` to “latest PyPI” while `torch_xla` stays on another build**.

**Override version** (keep both equal): `TORCH_VER=2.8.0 ./install_tpu_deps.sh`

If you only need to repair an existing venv:

```bash
source ~/pt-xla-env/bin/activate   # if you use a venv
cd /path/to/tpu_run
chmod +x repair_tpu_torch_abi.sh
./repair_tpu_torch_abi.sh
```

After any stray **`pip install torch ...`** or **`pip install -U torch`**, run `./repair_tpu_torch_abi.sh` or re-run `./install_tpu_deps.sh`.

### Sanity checks on the VM

```bash
python3 -m pip show torch torch-xla | grep -E '^Name:|^Version:'
python3 -c "import torch; print(torch.__file__)"
```

Both packages should sit under the same environment (e.g. `~/pt-xla-env/...`) and **versions should match** the pin (e.g. `2.9.0`).

---

## Shell scripts and Windows line endings

If `./install_tpu_deps.sh` fails with:

`/usr/bin/env: 'bash\r': No such file or directory`

the `.sh` files have CRLF line endings. On the VM:

```bash
sed -i 's/\r$//' install_tpu_deps.sh run_all.sh
chmod +x install_tpu_deps.sh run_all.sh
```

---

## Virtual environment (optional)

If `python3 -m venv` fails with **ensurepip / python3-venv** missing:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
# or explicitly on some images:
sudo apt-get install -y python3.10-venv
```

Then create the venv, **`source .../bin/activate`**, and either run **`./install_tpu_deps.sh`** from `tpu_run` (recommended) or mirror its steps: `requirements-tpu.txt`, then **uninstall** torch stack, then **pinned** `torch==X` + `torch_xla[tpu]==X` with **both** Google `-f` URLs (see `install_tpu_deps.sh`). Running **`python3 train_tpu.py` without activating the venv** uses system Python and old wheels — always **`which python3`** after `activate`.

---

## Outputs and GCS

**Default (local)**

- Checkpoints: `./outputs/checkpoints`
- Final model: `./outputs/final_model`

**GCS** (optional): pass `--gcs_bucket YOUR_BUCKET` (no `gs://` prefix). Checkpoints go under `gs://YOUR_BUCKET/models/...` per `train_tpu.py`. The VM’s **service account** needs permission to write objects (and to create the bucket if it does not exist). See `train_tpu.py` for `resolve_gcs_output_dir` behavior.

---

## Verify PyTorch/XLA

```bash
export PJRT_DEVICE=TPU
python3 -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices('TPU'))"
```

---

## Troubleshooting: `Failed to get global TPU topology`

That message means PJRT could not discover your TPU layout — **imports worked**, but the runtime cannot talk to the chips.

**1. Single-host VM (e.g. one VM with 8 cores)** — usual checks:

```bash
echo "$PJRT_DEVICE"          # should be TPU (or empty; train_tpu.py defaults it)
env | grep -i XRT            # should be empty; if XRT_TPU_CONFIG is set: unset XRT_TPU_CONFIG
pip show libtpu torch-xla    # libtpu should be present with torch_xla[tpu]
```

Then reinstall the pinned stack if needed: `./repair_tpu_torch_abi.sh` (venv activated).

**2. Multi-host TPU slice** (several VMs; SSH hostnames like `…-w-0`, `…-w-1`, …) — running **`python3 train_tpu.py` only on worker 0** often triggers this. Google expects the **same command on every worker at once**, e.g.:

```bash
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --zone="${ZONE}" --project="${PROJECT_ID}" --worker=all \
  --command="PJRT_DEVICE=TPU bash -lc 'source \$HOME/pt-xla-env/bin/activate && cd \$HOME/Model-Training/.../tpu_run && python3 train_tpu.py --data_csv ./data/merged_normalized.csv'"
```

Adjust paths so **every worker** has the venv, repo, and CSV (copy/fuse to all nodes). Details: [Run PyTorch code on TPU slices](https://cloud.google.com/tpu/docs/pytorch-pods).

**Note:** Hugging Face `Trainer` on **multi-host** TPUs usually needs an SPMD / multi-process setup beyond a single interactive shell; scaling across many VMs may require extra changes (see PyTorch/XLA distributed docs). Starting on a **single** TPU VM (e.g. v4-8) avoids this class of issues.

---

## Useful `train_tpu.py` flags

- `--data_csv` — required path to CSV.
- `--gcs_bucket` / `--gcs_location` — optional GCS layout.
- `--output_dir` — local checkpoint dir when **not** using `--gcs_bucket`.
- `--subset_rows N` — smoke test on first N rows (`0` = full data).
- `--no_resume` — do not resume from last checkpoint.

Full list: `python3 train_tpu.py --help`.
