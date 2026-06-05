#!/usr/bin/env python3
"""Prepare a simple caption-file image dataset for Musubi Z-Image LoRA training."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image, ImageOps


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directory containing source images.")
    parser.add_argument("--output", default=Path("data/my_lora"), type=Path, help="Prepared dataset directory.")
    parser.add_argument("--trigger", default="nytw_person", help="LoRA trigger word.")
    parser.add_argument("--class-caption", default="a photo of a person", help="Base caption after trigger.")
    parser.add_argument(
        "--resolution",
        default=int(os.environ.get("TRAIN_RESOLUTION", "768")),
        type=int,
        help="Square training resolution for dataset TOML.",
    )
    parser.add_argument("--min-images", default=6, type=int, help="Minimum accepted image count.")
    parser.add_argument("--max-images", default=16, type=int, help="Maximum images to copy.")
    parser.add_argument("--max-edge", default=1600, type=int, help="Resize copied images so the long edge is at most this.")
    return parser.parse_args()


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)


def save_image(src: Path, dst: Path, max_edge: int) -> None:
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        scale = min(1.0, max_edge / max(width, height))
        if scale < 1.0:
            image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
        image.save(dst, quality=95)


def write_dataset_toml(path: Path, image_dir: Path, cache_dir: Path, resolution: int) -> None:
    path.write_text(
        "\n".join(
            [
                "[general]",
                f"resolution = [{resolution}, {resolution}]",
                'caption_extension = ".txt"',
                "batch_size = 1",
                "enable_bucket = true",
                "bucket_no_upscale = false",
                "",
                "[[datasets]]",
                f'image_directory = "{image_dir.resolve()}"',
                f'cache_directory = "{cache_dir.resolve()}"',
                "num_repeats = 1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    images_dir = output_dir / "images"
    cache_dir = output_dir / "cache"

    if not source_dir.exists():
        raise SystemExit(f"Input directory does not exist: {source_dir}")

    files = image_files(source_dir)
    if len(files) < args.min_images:
        raise SystemExit(
            f"Need at least {args.min_images} images, found {len(files)} in {source_dir}. "
            "Use clearer individual photos or use the prepared Drive dataset."
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    caption = f"{args.trigger}, {args.class_caption}"
    selected = files[: args.max_images]

    for index, src in enumerate(selected, start=1):
        stem = f"{args.trigger}_{index:03d}"
        image_path = images_dir / f"{stem}.jpg"
        caption_path = images_dir / f"{stem}.txt"
        save_image(src, image_path, args.max_edge)
        caption_path.write_text(caption + "\n", encoding="utf-8")

    write_dataset_toml(output_dir / "dataset.toml", images_dir, cache_dir, args.resolution)

    print(f"Prepared {len(selected)} images")
    print(f"Dataset: {output_dir}")
    print(f"Config:  {output_dir / 'dataset.toml'}")
    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
