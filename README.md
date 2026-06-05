# Fine-Tune Image Model

NY Tech Week workshop launchable for training a small image LoRA with Z-Image and Musubi Tuner on NVIDIA Brev.

This repo is meant to be boring to operate: launch the Brev workspace, open the notebook, run cells top to bottom, and leave with a trained LoRA plus a small image gallery.

## What The Launchable Does

The default notebook:

1. Sets up Musubi Tuner and checks the GPU.
2. Downloads the Z-Image model weights.
3. Pulls a prepared person dataset from a public Google Drive folder.
4. Checks that every image has a matching caption.
5. Trains a LoRA.
6. Generates a few sample images.
7. Creates a ZIP download with checkpoints, prompts, and generated images.

Default dataset:

```text
DATASET_URL=https://drive.google.com/drive/folders/15sox-p1gwyb5lD52p5bfx-vkabVu_sRV?usp=sharing
TRIGGER=jalen
OUTPUT_NAME=jalen_zimage
```

## Brev Launchable Settings

Use the Brev Console Launchables flow with:

- **Code source:** public Git repo containing this folder
- **Mode:** VM Mode
- **GPU:** 1x A100, preferably 80GB
- **Disk:** 200GB minimum, 300GB safer
- **Jupyter:** enabled
- **Start notebook:** `notebooks/01_train_personal_lora.ipynb`
- **Pre-build hook:** `bash preBuild.bash`
- **Post-build hook:** `bash postBuild.bash`

The post-build hook installs Python dependencies and Musubi Tuner. It does not download the Z-Image model by default because the model cache is large. The notebook downloads the model on first run.

## Replacing The Dataset

Put a public Google Drive folder or ZIP in this shape:

```text
dataset/
  images/
    image_001.jpg
    image_001.txt
    image_002.jpg
    image_002.txt
    ...
  sample_prompts.txt
```

Then change these values in `variables.env`:

```bash
DATASET_URL=https://drive.google.com/drive/folders/...
TRIGGER=my_subject
OUTPUT_NAME=my_subject_zimage
PROMPT_FILE=data/dataset/sample_prompts.txt
```

For a person LoRA, use at least 12 images. The smoother workshop target is 20-24 images. Use mostly clear face images with varied angles, lighting, and clothing. Avoid too many duplicate shots, group shots, heavy filters, tiny faces, sunglasses in every photo, or images where the person is far from camera.

Every image needs a same-name `.txt` caption. Keep captions short and put the trigger first:

```text
my_subject, a photo of a person, close-up headshot, wearing a black jacket
my_subject, a photo of a person, chest-up portrait outdoors, natural lighting
my_subject, a photo of a person, full-body photo, casual outfit, city street
```

Also update `sample_prompts.txt` to use the same trigger:

```text
my_subject, a photo of a person, modern professional headshot, clean background, soft studio lighting --w 768 --h 768 --s 24 --d 3301 --l 1
```

The notebook validates image count, missing captions, and missing trigger words before training.

## Local Layout

```text
notebooks/
  01_train_personal_lora.ipynb
scripts/
  setup_brev.sh
  download_models.sh
  download_workshop_dataset.py
  validate_dataset.py
  train_zimage_lora.sh
  generate_zimage_gallery.sh
configs/
  sample_prompts.txt
variables.env
```

## Notes

- Keep the base model out of git.
- Keep attendee datasets out of git.
- Only use datasets you have permission to train on.
- If the model already exists in the workspace, the notebook skips downloading it again.
