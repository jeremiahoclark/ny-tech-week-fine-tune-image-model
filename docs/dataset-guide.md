# Dataset Guide

Use this when you want to replace the default Drive dataset with a different person.

## Folder Shape

The downloader accepts a public Google Drive folder or ZIP. The easiest shape is:

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

The folder can be named `dataset`, or it can be inside another Drive folder. The notebook searches for the first folder that has `images/` and a dataset layout, then copies it into:

```text
data/dataset/
```

## Image Count

Minimum:

```text
12 images
```

Better workshop target:

```text
20-24 images
```

Useful image mix:

- clear face visible
- close-up, chest-up, and a few full-body photos
- varied angles and lighting
- different outfits or settings
- one person per image

Avoid:

- lots of near-duplicate photos
- tiny face or far-away action shots
- heavy sunglasses in most images
- group photos
- screenshots with text overlays
- blurry or low-resolution images

## Captions

Every image needs a same-name `.txt` file:

```text
image_001.jpg
image_001.txt
```

Put the trigger word first. Keep the rest short and literal.

Good examples:

```text
my_subject, a photo of a person, close-up headshot, wearing a black jacket
my_subject, a photo of a person, chest-up portrait outdoors, natural lighting
my_subject, a photo of a person, full-body photo, casual outfit, city street
```

Do not write long captions full of things you want to generate later. Captions describe the training image. Prompts describe the output you want after training.

## Trigger Word

The trigger is the handle the LoRA learns for the subject.

For the default dataset:

```text
jalen
```

For a replacement dataset, use something unlikely to appear accidentally:

```text
my_subject
nytw_person
subject_alpha
```

Then update all three places:

```bash
TRIGGER=my_subject
OUTPUT_NAME=my_subject_zimage
```

Each caption:

```text
my_subject, a photo of a person, ...
```

And each line in `sample_prompts.txt`:

```text
my_subject, a photo of a person, modern professional headshot, clean background --w 768 --h 768 --s 24 --d 3301 --l 1
```

## Google Drive Sharing

Share the Drive folder so anyone with the link can view it.

Paste the folder URL into `variables.env`:

```bash
DATASET_URL=https://drive.google.com/drive/folders/...
```

The notebook uses `gdown` to download the folder into the Brev workspace. If Drive blocks a large folder download, zip the `dataset/` folder, upload the ZIP, and use the ZIP share link instead.

## What The Notebook Checks

Before training, the notebook runs:

```bash
python scripts/validate_dataset.py --dataset data/dataset --trigger "$TRIGGER"
```

That catches:

- too few images
- missing `.txt` captions
- empty captions
- captions that do not contain the trigger word
