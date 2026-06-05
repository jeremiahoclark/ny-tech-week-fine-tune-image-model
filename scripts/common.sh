#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MUSUBI_DIR="${MUSUBI_DIR:-"$ROOT_DIR/vendor/musubi-tuner"}"
MODEL_DIR="${MODEL_DIR:-"$ROOT_DIR/models/z-image"}"
OUTPUT_ROOT="${OUTPUT_ROOT:-"$ROOT_DIR/outputs"}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PATH="$HOME/.local/bin:$PATH"

setup_cache_env() {
  export HF_HOME="${HF_HOME:-"$ROOT_DIR/.cache/hf"}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-"$HF_HOME/hub"}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-"$HF_HOME/transformers"}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-"$ROOT_DIR/.cache/xdg"}"
  export TORCH_HOME="${TORCH_HOME:-"$ROOT_DIR/.cache/torch"}"
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME"
}

first_match() {
  local pattern
  for pattern in "$@"; do
    local found
    found="$(compgen -G "$pattern" | sort | head -n 1 || true)"
    if [[ -n "$found" ]]; then
      printf "%s\n" "$found"
      return 0
    fi
  done
  return 1
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    return 1
  fi
}

resolve_zimage_paths() {
  ZIMAGE_DIT="${ZIMAGE_DIT:-"$(first_match \
    "$MODEL_DIR/transformer/*00001-of-*.safetensors" \
    "$MODEL_DIR/transformer/*.safetensors" \
    "$MODEL_DIR/*dit*.safetensors" \
    "$MODEL_DIR/*transformer*.safetensors" || true)"}"

  ZIMAGE_VAE="${ZIMAGE_VAE:-"$(first_match \
    "$MODEL_DIR/vae/*.safetensors" \
    "$MODEL_DIR/*vae*.safetensors" || true)"}"

  ZIMAGE_TEXT_ENCODER="${ZIMAGE_TEXT_ENCODER:-"$(first_match \
    "$MODEL_DIR/text_encoder/*00001-of-*.safetensors" \
    "$MODEL_DIR/text_encoder/*.safetensors" \
    "$MODEL_DIR/*text*encoder*.safetensors" || true)"}"

  if [[ -z "${ZIMAGE_DIT:-}" || -z "${ZIMAGE_VAE:-}" || -z "${ZIMAGE_TEXT_ENCODER:-}" ]]; then
    cat >&2 <<EOF
Could not resolve all Z-Image model paths.

Run:
  bash scripts/download_models.sh

Or set these manually:
  export ZIMAGE_DIT=/path/to/first-transformer-shard.safetensors
  export ZIMAGE_VAE=/path/to/vae.safetensors
  export ZIMAGE_TEXT_ENCODER=/path/to/first-text-encoder-shard.safetensors
EOF
    return 1
  fi

  export ZIMAGE_DIT ZIMAGE_VAE ZIMAGE_TEXT_ENCODER
}

print_gpu_summary() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  else
    echo "nvidia-smi not found"
  fi
}

setup_cache_env
