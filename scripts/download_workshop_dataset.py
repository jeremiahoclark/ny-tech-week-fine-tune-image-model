#!/usr/bin/env python3
"""Download or copy a workshop dataset from Google Drive and make it portable."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


DATASETS = {
    "custom": {
        "env": "DATASET_URL",
        "output": Path("data/dataset"),
        "archive": None,
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="custom")
    parser.add_argument(
        "--url",
        help=(
            "Google Drive file/folder share URL. Defaults to DATASET_URL."
        ),
    )
    parser.add_argument("--output", type=Path, help="Output dataset directory.")
    parser.add_argument("--archive", type=Path, help="Existing local archive to unpack instead of downloading.")
    parser.add_argument("--drive-folder", action="store_true", help="Treat --url as a Google Drive folder URL.")
    parser.add_argument("--force", action="store_true", help="Replace output directory if it already exists.")
    parser.add_argument("--resolution", type=int, default=int(os.environ.get("TRAIN_RESOLUTION", "768")))
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def import_gdown():
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "gdown is required to download Google Drive files or folders. Install it with:\n"
            "  python -m pip install gdown\n"
        ) from exc
    return gdown


def looks_like_drive_folder(url: str) -> bool:
    return "/drive/folders/" in url or "folders/" in url


def download_file_with_gdown(url: str, archive_path: Path) -> None:
    gdown = import_gdown()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(url=url, output=str(archive_path), quiet=False, fuzzy=True)
    if not result or not archive_path.exists() or archive_path.stat().st_size == 0:
        raise SystemExit(f"Download failed or produced an empty file: {url}")


def download_folder_with_gdown(url: str, target_dir: Path) -> None:
    gdown = import_gdown()
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = gdown.download_folder(url=url, output=str(target_dir), quiet=False)
    except TypeError:
        result = gdown.download_folder(url, output=str(target_dir), quiet=False)
    if not result:
        raise SystemExit(f"Drive folder download failed: {url}")


def has_images_dir(path: Path) -> bool:
    images_dir = path / "images"
    if not images_dir.is_dir():
        return False
    return any(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for p in images_dir.iterdir())


def find_dataset_root(extract_dir: Path) -> Path:
    candidates = [p for p in extract_dir.rglob("dataset.toml") if (p.parent / "images").is_dir()]
    if not candidates:
        image_candidates = [p for p in extract_dir.rglob("images") if has_images_dir(p.parent)]
        if not image_candidates:
            raise SystemExit(
                "No dataset root found. Expected either:\n"
                "  dataset/dataset.toml plus dataset/images/\n"
                "or:\n"
                "  dataset/images/ with image files and same-name .txt captions\n"
                f"Searched under: {extract_dir}"
            )
        return sorted((p.parent for p in image_candidates), key=lambda p: len(p.parts))[0]
    if len(candidates) > 1:
        candidates = sorted(candidates, key=lambda p: len(p.parts))
    return candidates[0].parent


def write_dataset_toml(dataset_dir: Path, resolution: int) -> None:
    images_dir = dataset_dir / "images"
    cache_dir = dataset_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset.toml").write_text(
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
                f'image_directory = "{images_dir.resolve()}"',
                f'cache_directory = "{cache_dir.resolve()}"',
                "num_repeats = 1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = repo_root()
    defaults = DATASETS[args.dataset]
    output = (args.output or defaults["output"])
    output = output if output.is_absolute() else root / output
    env_url = os.environ.get("DATASET_URL") or os.environ.get(defaults["env"])
    url = args.url or env_url
    archive = args.archive or (None if url else defaults["archive"])
    archive = archive if archive is None or archive.is_absolute() else root / archive

    if output.exists() and (output / "dataset.toml").exists() and not args.force:
        write_dataset_toml(output, args.resolution)
        print(f"Dataset already present: {output}")
        print(f"Config rewritten: {output / 'dataset.toml'}")
        return

    if output.exists():
        if not args.force:
            raise SystemExit(f"Output exists. Re-run with --force to replace: {output}")
        shutil.rmtree(output)

    with tempfile.TemporaryDirectory(prefix=f"nytw-{args.dataset}-") as tmp:
        tmp_dir = Path(tmp)
        extract_dir = tmp_dir / "extract"
        extract_dir.mkdir(parents=True)

        if archive and archive.exists():
            print(f"Using local archive: {archive}")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extract_dir)
        else:
            if not url:
                env_hint = defaults["env"]
                raise SystemExit(
                    f"No local archive found and no URL provided. Set DATASET_URL or {env_hint}, or pass --url."
                )
            if args.drive_folder or looks_like_drive_folder(url):
                print(f"Downloading {args.dataset} dataset folder from Google Drive...")
                download_folder_with_gdown(url, extract_dir)
            else:
                if archive is None:
                    archive = tmp_dir / "dataset.zip"
                print(f"Downloading {args.dataset} dataset archive from Google Drive...")
                download_file_with_gdown(url, archive)
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(extract_dir)

        dataset_root = find_dataset_root(extract_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dataset_root), str(output))

    write_dataset_toml(output, args.resolution)
    image_count = sum(1 for p in (output / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    caption_count = sum(1 for p in (output / "images").iterdir() if p.suffix.lower() == ".txt")
    print(f"Dataset ready: {output}")
    print(f"Images: {image_count}")
    print(f"Captions: {caption_count}")
    print(f"Config: {output / 'dataset.toml'}")


if __name__ == "__main__":
    main()
