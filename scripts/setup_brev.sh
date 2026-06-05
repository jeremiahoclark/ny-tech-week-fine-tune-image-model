#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

MUSUBI_REF="${MUSUBI_REF:-v0.3.0}"
MUSUBI_CUDA_EXTRA="${MUSUBI_CUDA_EXTRA:-cu128}"

echo "== GPU =="
print_gpu_summary || true

echo "== Python =="
"$PYTHON_BIN" --version

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  echo "pip is missing for $PYTHON_BIN; bootstrapping pip..."
  if "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1; then
    echo "pip installed with ensurepip."
  elif command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PYTHON_BIN" pip setuptools wheel
  else
    tmp_get_pip="$(mktemp /tmp/get-pip.XXXXXX.py)"
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$tmp_get_pip"
    "$PYTHON_BIN" "$tmp_get_pip"
    rm -f "$tmp_get_pip"
  fi
fi

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

echo "== Workshop Python packages =="
"$PYTHON_BIN" -m pip install --upgrade \
  hf_transfer \
  gdown \
  huggingface_hub \
  ipywidgets \
  jupyter-app-launcher \
  jupyterlab-git \
  matplotlib \
  pandas \
  pillow \
  voila

if [[ "$MUSUBI_CUDA_EXTRA" == "cu124" ]]; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
  TORCH_SPEC="torch>=2.5.1"
  TORCHVISION_SPEC="torchvision>=0.20.1"
elif [[ "$MUSUBI_CUDA_EXTRA" == "cu130" ]]; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
  TORCH_SPEC="torch>=2.9.1"
  TORCHVISION_SPEC="torchvision>=0.24.1"
else
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
  TORCH_SPEC="torch>=2.7.1"
  TORCHVISION_SPEC="torchvision>=0.22.1"
fi

echo "== PyTorch CUDA wheel: $MUSUBI_CUDA_EXTRA =="
"$PYTHON_BIN" -m pip install --upgrade \
  --index-url "$TORCH_INDEX_URL" \
  --extra-index-url https://pypi.org/simple \
  "$TORCH_SPEC" "$TORCHVISION_SPEC"

if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
  echo "== Python requirements.txt =="
  "$PYTHON_BIN" -m pip install --upgrade -r "$ROOT_DIR/requirements.txt"
fi

echo "== Musubi Tuner =="
mkdir -p "$ROOT_DIR/vendor"
if [[ ! -d "$MUSUBI_DIR/.git" ]]; then
  git clone https://github.com/kohya-ss/musubi-tuner.git "$MUSUBI_DIR"
fi

git -C "$MUSUBI_DIR" fetch --depth 1 origin "$MUSUBI_REF"
git -C "$MUSUBI_DIR" checkout FETCH_HEAD
"$PYTHON_BIN" -m pip install -e "$MUSUBI_DIR"

echo "== Accelerate default config =="
mkdir -p "$ROOT_DIR/configs"
cat > "$ROOT_DIR/configs/accelerate-single-gpu.yaml" <<'YAML'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: "NO"
downcast_bf16: "no"
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML

echo "Setup complete."
echo "Next: bash scripts/download_models.sh"
