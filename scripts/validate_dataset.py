#!/usr/bin/env python3
"""Validate the image/caption dataset before a workshop LoRA run."""

from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--min-images", type=int, default=12)
    parser.add_argument("--recommended-images", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset
    images_dir = dataset_dir / "images"
    trigger = args.trigger.strip()

    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset folder does not exist: {dataset_dir}")
    if not images_dir.is_dir():
        raise SystemExit(f"Expected image folder: {images_dir}")

    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(image_files) < args.min_images:
        raise SystemExit(
            f"Found {len(image_files)} images. Add at least {args.min_images}; "
            f"{args.recommended_images}-24 is better for this workshop."
        )

    missing_captions: list[Path] = []
    missing_trigger: list[Path] = []
    empty_captions: list[Path] = []

    for image_path in image_files:
        caption_path = image_path.with_suffix(".txt")
        if not caption_path.exists():
            missing_captions.append(caption_path)
            continue
        caption = caption_path.read_text(encoding="utf-8").strip()
        if not caption:
            empty_captions.append(caption_path)
        if trigger and trigger.lower() not in caption.lower():
            missing_trigger.append(caption_path)

    if missing_captions:
        preview = "\n".join(f"  - {p}" for p in missing_captions[:10])
        raise SystemExit(f"Missing same-name caption files:\n{preview}")
    if empty_captions:
        preview = "\n".join(f"  - {p}" for p in empty_captions[:10])
        raise SystemExit(f"Empty caption files:\n{preview}")
    if missing_trigger:
        preview = "\n".join(f"  - {p}" for p in missing_trigger[:10])
        raise SystemExit(f"Caption files missing trigger word {trigger!r}:\n{preview}")

    print("Dataset check passed.")
    print(f"Dataset: {dataset_dir}")
    print(f"Images: {len(image_files)}")
    print(f"Captions: {len(image_files)}")
    print(f"Trigger: {trigger}")
    if len(image_files) < args.recommended_images:
        print(f"Note: {args.recommended_images}-24 images usually gives a better identity LoRA.")


if __name__ == "__main__":
    main()
