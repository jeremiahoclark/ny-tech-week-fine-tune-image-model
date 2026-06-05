#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

HF_MODEL_REPO="${HF_MODEL_REPO:-Tongyi-MAI/Z-Image}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p "$MODEL_DIR"
"$PYTHON_BIN" -m pip install --upgrade "huggingface_hub==0.34.3" hf_transfer

echo "Downloading $HF_MODEL_REPO to $MODEL_DIR"
huggingface-cli download "$HF_MODEL_REPO" \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False

echo "Resolving model paths..."
resolve_zimage_paths

cat <<EOF
Model download complete.

ZIMAGE_DIT=$ZIMAGE_DIT
ZIMAGE_VAE=$ZIMAGE_VAE
ZIMAGE_TEXT_ENCODER=$ZIMAGE_TEXT_ENCODER
EOF
