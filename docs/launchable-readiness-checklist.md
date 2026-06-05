# Launchable Readiness Checklist

Checked on 2026-06-05 against NVIDIA Brev docs:

- Brev Launchables docs: https://docs.nvidia.com/brev/concepts/launchables
- Brev Jupyter docs: https://docs.nvidia.com/brev/guides/development-tools/jupyter-notebooks

## Brev Requirements

- [x] Code can come from a public Git repository.
- [x] VM Mode is appropriate for this project because we use bash setup scripts, apt, pip, and full VM access.
- [x] Jupyter should be enabled so attendees can open the notebook directly in JupyterLab.
- [x] Compute should be 1x A100, preferably 80GB, with at least 200GB disk.
- [x] Launchable visibility should be `Anyone with the link` for the workshop, unless intentionally publishing to the community directory.

## Repository Shape

- [x] Repo root contains `README.md`.
- [x] Repo root contains `notebooks/01_train_personal_lora.ipynb`.
- [x] Repo root contains `preBuild.bash`.
- [x] Repo root contains `postBuild.bash`.
- [x] Repo root contains `variables.env` with public workshop defaults.
- [x] Repo root contains setup/training/generation scripts under `scripts/`.
- [x] Repo excludes model weights, outputs, generated datasets, and `.safetensors` files.
- [x] No Tuxemon/private iteration files remain in the staged public repo folder.

## Notebook Flow

- [x] Loads `variables.env`.
- [x] Checks GPU with `nvidia-smi`.
- [x] Installs or verifies Musubi Tuner.
- [x] Downloads Z-Image if model files are missing.
- [x] Downloads the public Google Drive dataset.
- [x] Rewrites `dataset.toml` with Brev-local paths.
- [x] Validates image count, captions, and trigger word.
- [x] Trains LoRA with `MAX_TRAIN_STEPS=800`, `NETWORK_DIM=32`, `NETWORK_ALPHA=32`.
- [x] Generates image gallery from `data/dataset/sample_prompts.txt`.
- [x] Creates a downloadable ZIP with checkpoints, generated images, captions, prompts, and config.

## Dataset Checks

- [x] Public Drive folder is accessible without Google auth.
- [x] Public Drive folder downloads through `gdown`.
- [x] Downloaded dataset has 20 images.
- [x] Downloaded dataset has 20 matching captions.
- [x] Captions include trigger `jalen`.
- [x] `sample_prompts.txt` is present in the Drive dataset.

## Launchable Console Settings

Use these in the Brev Console wizard:

```text
Code source: Git Repository
Repository: https://github.com/jeremiahoclark/ny-tech-week-fine-tune-image-model
Runtime: VM Mode
Setup script: bash postBuild.bash
Jupyter Notebook Experience: Yes
GPU: 1x A100
Disk: 200GB minimum, 300GB safer
View access: Anyone with the link
Start notebook: notebooks/01_train_personal_lora.ipynb
```

If the wizard has separate pre-build and post-build hooks, use:

```text
Pre-build: bash preBuild.bash
Post-build: bash postBuild.bash
```

If it has only one setup script field, use:

```bash
bash preBuild.bash && bash postBuild.bash
```

## Final Smoke Test

After creating the Launchable, deploy it once yourself and confirm:

- [ ] Jupyter opens directly to the notebook.
- [ ] `nvidia-smi` shows an A100.
- [ ] `bash scripts/download_models.sh` completes.
- [ ] Dataset download completes without Google auth.
- [ ] Dataset validator reports 20 images and 20 captions.
- [ ] Training produces `outputs/lora/jalen_zimage-latest.safetensors`.
- [ ] Generation produces images in `outputs/gallery`.
- [ ] Download button saves a ZIP locally.
