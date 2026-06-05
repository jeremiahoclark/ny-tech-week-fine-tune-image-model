#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

DATASET_CONFIG="${1:-"$ROOT_DIR/data/my_lora/dataset.toml"}"
OUTPUT_DIR="${2:-"$OUTPUT_ROOT/lora"}"
OUTPUT_NAME="${OUTPUT_NAME:-nytw_person_zimage}"
SAMPLE_PROMPTS="${SAMPLE_PROMPTS:-"$ROOT_DIR/configs/sample_prompts.txt"}"

MAX_TRAIN_EPOCHS="${MAX_TRAIN_EPOCHS:-3}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
NETWORK_DIM="${NETWORK_DIM:-16}"
NETWORK_ALPHA="${NETWORK_ALPHA:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SEED="${SEED:-42}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-4}"

resolve_zimage_paths
require_file "$DATASET_CONFIG" "dataset config"
require_file "$MUSUBI_DIR/src/musubi_tuner/zimage_train_network.py" "Musubi Z-Image trainer"

if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" == "1" ]] && ! "$PYTHON_BIN" -c "import hf_transfer" >/dev/null 2>&1; then
  echo "hf_transfer is not installed; disabling HF_HUB_ENABLE_HF_TRANSFER for this training run."
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi

mkdir -p "$OUTPUT_DIR"

echo "== Cache latents =="
"$PYTHON_BIN" "$MUSUBI_DIR/src/musubi_tuner/zimage_cache_latents.py" \
  --dataset_config "$DATASET_CONFIG" \
  --vae "$ZIMAGE_VAE"

echo "== Cache text encoder outputs =="
"$PYTHON_BIN" "$MUSUBI_DIR/src/musubi_tuner/zimage_cache_text_encoder_outputs.py" \
  --dataset_config "$DATASET_CONFIG" \
  --text_encoder "$ZIMAGE_TEXT_ENCODER" \
  --batch_size "$TEXT_BATCH_SIZE"

echo "== Train LoRA =="
cmd=(
  accelerate launch
  --config_file "$ROOT_DIR/configs/accelerate-single-gpu.yaml"
  --num_cpu_threads_per_process 1
  --mixed_precision bf16
  "$MUSUBI_DIR/src/musubi_tuner/zimage_train_network.py"
  --dit "$ZIMAGE_DIT"
  --vae "$ZIMAGE_VAE"
  --text_encoder "$ZIMAGE_TEXT_ENCODER"
  --dataset_config "$DATASET_CONFIG"
  --sdpa
  --mixed_precision bf16
  --timestep_sampling shift
  --weighting_scheme none
  --discrete_flow_shift 2.0
  --optimizer_type adamw8bit
  --learning_rate "$LEARNING_RATE"
  --gradient_checkpointing
  --max_data_loader_n_workers 2
  --persistent_data_loader_workers
  --network_module networks.lora_zimage
  --network_dim "$NETWORK_DIM"
  --network_alpha "$NETWORK_ALPHA"
  --save_every_n_epochs 1
  --seed "$SEED"
  --output_dir "$OUTPUT_DIR"
  --output_name "$OUTPUT_NAME"
)

if [[ -n "$MAX_TRAIN_STEPS" ]]; then
  cmd+=(--max_train_steps "$MAX_TRAIN_STEPS")
else
  cmd+=(--max_train_epochs "$MAX_TRAIN_EPOCHS")
fi

if [[ -n "${BLOCKS_TO_SWAP:-}" ]]; then
  cmd+=(--blocks_to_swap "$BLOCKS_TO_SWAP")
fi

if [[ -f "$SAMPLE_PROMPTS" && "${ENABLE_TRAINING_SAMPLES:-0}" == "1" ]]; then
  cmd+=(--sample_prompts "$SAMPLE_PROMPTS" --sample_every_n_epochs 1)
fi

"${cmd[@]}"

latest_lora="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name "$OUTPUT_NAME*.safetensors" -print | sort | tail -n 1)"
if [[ -n "$latest_lora" ]]; then
  (
    cd "$OUTPUT_DIR"
    ln -sfn "$(basename "$latest_lora")" "$OUTPUT_NAME-latest.safetensors"
  )
  echo "Latest LoRA symlink: $OUTPUT_DIR/$OUTPUT_NAME-latest.safetensors"
fi

echo "Training complete."
echo "LoRA output directory: $OUTPUT_DIR"
