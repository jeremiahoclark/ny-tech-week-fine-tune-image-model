# Brev Launchable Setup

This is the production checklist for turning this folder into the attendee-facing Brev Launchable.

## Launchable Shape

Create the Launchable from a public Git repository containing this folder as the repo root.

Recommended settings:

- **Runtime:** VM Mode
- **GPU:** 1x A100, preferably 80GB
- **Disk:** 200GB minimum, 300GB safer
- **Jupyter:** enabled
- **Start notebook:** `notebooks/01_train_personal_lora.ipynb`
- **Pre-build hook:** `bash preBuild.bash`
- **Post-build hook:** `bash postBuild.bash`
- **Visibility:** link sharing for workshop attendees

The default dataset is pulled from the public Google Drive folder in `variables.env`.

## What Build Does

`preBuild.bash` installs apt packages, Git LFS, and shell defaults.

`postBuild.bash` installs Python packages, PyTorch, Musubi Tuner, and the Accelerate config.

The build does **not** download the Z-Image model by default:

```bash
DOWNLOAD_MODELS_ON_BUILD=0
```

That keeps the launchable build lighter. The notebook downloads the model on the first run. If you want the model downloaded during build, set:

```bash
DOWNLOAD_MODELS_ON_BUILD=1
```

Only do that after testing the build time and disk usage.

## Dataset Defaults

Current workshop defaults:

```bash
WORKSHOP_DATASET=custom
DATASET_URL=https://drive.google.com/drive/folders/15sox-p1gwyb5lD52p5bfx-vkabVu_sRV?usp=sharing
TRIGGER=jalen
OUTPUT_NAME=jalen_zimage
PROMPT_FILE=data/dataset/sample_prompts.txt
```

The Drive folder should contain:

```text
dataset/
  images/
    image_001.jpg
    image_001.txt
    ...
  sample_prompts.txt
```

The notebook rewrites `dataset.toml` after download so Musubi sees the correct local Brev paths.

## Training Defaults

The Jalen run that looked best used the stronger person-LoRA settings:

```bash
TRAIN_RESOLUTION=768
MAX_TRAIN_STEPS=800
NETWORK_DIM=32
NETWORK_ALPHA=32
LEARNING_RATE=1e-4
INFER_STEPS=24
```

For a one-hour room, do not keep pushing steps live. If quality is weak, use a better dataset or a saved checkpoint rather than improvising longer training.

## Pre-Event Smoke Test

After publishing the repo and creating the Launchable, start a fresh workspace and run:

```bash
bash postBuild.bash
bash scripts/download_models.sh
python scripts/download_workshop_dataset.py \
  --dataset custom \
  --url "$DATASET_URL" \
  --output data/dataset \
  --force
python scripts/validate_dataset.py \
  --dataset data/dataset \
  --trigger "$TRIGGER" \
  --min-images "$MIN_IMAGES"
bash scripts/train_zimage_lora.sh \
  data/dataset/dataset.toml \
  outputs/lora
bash scripts/generate_zimage_gallery.sh \
  outputs/lora/jalen_zimage-latest.safetensors \
  data/dataset/sample_prompts.txt \
  outputs/gallery
```

Then verify:

- Jupyter opens directly to the notebook.
- `nvidia-smi` shows the expected A100.
- Dataset download succeeds without Google auth.
- The validator reports at least 12 images and matching captions.
- Training writes `outputs/lora/jalen_zimage-latest.safetensors`.
- Generation writes images into `outputs/gallery`.
- The final notebook cell creates a downloadable ZIP.

## Replacing The Dataset

Use [dataset-guide.md](dataset-guide.md).

Short version:

1. Create a Drive folder or ZIP with `dataset/images`.
2. Use at least 12 images, ideally 20-24.
3. Add same-name `.txt` captions for every image.
4. Put the trigger word first in every caption.
5. Update `DATASET_URL`, `TRIGGER`, `OUTPUT_NAME`, and `sample_prompts.txt`.
6. Run the notebook top to bottom.

## Source References

- Brev Launchables: https://docs.nvidia.com/brev/concepts/launchables
- Brev Jupyter: https://docs.nvidia.com/brev/guides/development-tools/jupyter-notebooks
- Musubi Tuner: https://github.com/kohya-ss/musubi-tuner
- Musubi Z-Image docs: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/zimage.md
- Z-Image model: https://huggingface.co/Tongyi-MAI/Z-Image
