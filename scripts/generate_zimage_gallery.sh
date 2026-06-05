#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

LORA_PATH="${1:-"$OUTPUT_ROOT/lora/nytw_person_zimage-latest.safetensors"}"
PROMPTS_FILE="${2:-"$ROOT_DIR/configs/sample_prompts.txt"}"
SAVE_PATH="${3:-"$OUTPUT_ROOT/gallery"}"

IMAGE_HEIGHT="${IMAGE_HEIGHT:-768}"
IMAGE_WIDTH="${IMAGE_WIDTH:-768}"
INFER_STEPS="${INFER_STEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
FLOW_SHIFT="${FLOW_SHIFT:-3.0}"
SEED="${SEED:-1234}"
LORA_MULTIPLIER="${LORA_MULTIPLIER:-1.0}"
ATTN_MODE="${ATTN_MODE:-torch}"

resolve_zimage_paths
require_file "$MUSUBI_DIR/src/musubi_tuner/zimage_generate_image.py" "Musubi Z-Image generator"
require_file "$PROMPTS_FILE" "prompts file"

if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" == "1" ]] && ! "$PYTHON_BIN" -c "import hf_transfer" >/dev/null 2>&1; then
  echo "hf_transfer is not installed; disabling HF_HUB_ENABLE_HF_TRANSFER for this generation run."
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi

mkdir -p "$SAVE_PATH"

cmd=(
  "$PYTHON_BIN" "$MUSUBI_DIR/src/musubi_tuner/zimage_generate_image.py"
  --dit "$ZIMAGE_DIT"
  --vae "$ZIMAGE_VAE"
  --text_encoder "$ZIMAGE_TEXT_ENCODER"
  --from_file "$PROMPTS_FILE"
  --image_size "$IMAGE_HEIGHT" "$IMAGE_WIDTH"
  --infer_steps "$INFER_STEPS"
  --flow_shift "$FLOW_SHIFT"
  --guidance_scale "$GUIDANCE_SCALE"
  --attn_mode "$ATTN_MODE"
  --save_path "$SAVE_PATH"
  --seed "$SEED"
)

if [[ -f "$LORA_PATH" ]]; then
  cmd+=(--lora_weight "$LORA_PATH" --lora_multiplier "$LORA_MULTIPLIER")
else
  echo "LoRA not found at $LORA_PATH; generating without LoRA."
fi

"${cmd[@]}"

echo "Gallery saved to $SAVE_PATH"
