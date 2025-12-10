#!/usr/bin/env python3
import argparse
import csv
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        if dst.exists():
            return
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def prepare_isic2018(root: Path) -> Path:
    """Prepare ISIC2018 Task 3 into imagefolder structure.

    Expects zip files named like:
      - ISIC2018_Task3_Training_Input.zip
      - ISIC2018_Task3_Training_GroundTruth.zip (contains CSV)
      - ISIC2018_Task3_Validation_Input.zip
      - ISIC2018_Task3_Validation_GroundTruth.zip
      - ISIC2018_Task3_Test_Input.zip
      - ISIC2018_Task3_Test_GroundTruth.zip

    Produces: root/imagefolder/{train,val,test}/{class}/image.jpg (symlinks)
    """
    root = root.resolve()
    out_root = root / "imagefolder"
    ensure_dir(out_root)

    # Unzip all archives into raw/ subdir
    raw = root / "raw"
    ensure_dir(raw)
    zips = list(root.glob("ISIC2018_Task3_*_Input.zip")) + list(root.glob("ISIC2018_Task3_*_GroundTruth.zip"))
    for zf in zips:
        target_dir = raw / zf.stem
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(target_dir)

    # Locate CSVs and image folders
    def find_csv(split: str) -> Path:
        base = raw / f"ISIC2018_Task3_{split}_GroundTruth"
        cand = list(base.glob("*.csv"))
        if not cand:
            # fallback: recursive search
            cand = [p for p in raw.rglob("*.csv") if f"Task3_{split}_GroundTruth" in p.name]
        if not cand:
            raise FileNotFoundError(f"ISIC2018 {split} ground truth CSV not found under {base}")
        return cand[0]

    def find_images(split: str) -> Path:
        # images are directly under this extracted dir (possibly with nested folder)
        base = raw / f"ISIC2018_Task3_{split}_Input"
        # some distributions nest one folder level; flatten by finding all images
        return base

    splits = ["Training", "Validation", "Test"]
    class_names = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]

    for split in splits:
        csv_path = find_csv(split)
        img_dir = find_images(split)
        # Build mapping: image_id -> class name
        mapping = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                # CSV may have first column as image name or explicit 'image'
                img_id = row.get('image') or row.get('Image') or row.get('lesion_id')
                if not img_id:
                    # Some files may have first unnamed column as image id
                    img_id = list(row.values())[0]
                label_name = None
                for cname in class_names:
                    v = row.get(cname, '')
                    try:
                        val = float(v or 0.0)
                    except ValueError:
                        val = 0.0
                    if val >= 1.0:
                        label_name = cname
                        break
                if label_name is None:
                    # If none marked 1, skip
                    continue
                mapping[img_id] = label_name

        # Link images
        of_split = out_root / ("train" if split=="Training" else ("val" if split=="Validation" else "test"))
        for cname in class_names:
            ensure_dir(of_split / cname)
        # Images in dir possibly nested; find all jpg/png
        for p in img_dir.rglob("*.jpg"):
            stem = p.stem
            cname = mapping.get(stem)
            if cname:
                dst = of_split / cname / p.name
                symlink_or_copy(p, dst)
        for p in img_dir.rglob("*.png"):
            stem = p.stem
            cname = mapping.get(stem)
            if cname:
                dst = of_split / cname / p.name
                symlink_or_copy(p, dst)

    return out_root


def prepare_odir(root: Path, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> Path:
    """Prepare ODIR into imagefolder structure by splitting existing class folders.

    Expects class subfolders under root (e.g., normal, diabetes, etc.).
    Produces: root/imagefolder/{train,val,test}/{class}/image.jpg (symlinks)
    """
    root = root.resolve()
    out_root = root / "imagefolder"
    ensure_dir(out_root)
    rng = random.Random(seed)

    # Detect classes by folders containing images
    classes = [d.name for d in root.iterdir() if d.is_dir() and d.name not in {"imagefolder", "raw"}]
    for split in ("train", "val", "test"):
        for c in classes:
            ensure_dir(out_root / split / c)

    for c in classes:
        images = []
        cdir = root / c
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            images.extend(list(cdir.glob(ext)))
        images.sort()
        if not images:
            continue
        rng.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        splits = (
            ("train", images[:n_train]),
            ("val", images[n_train:n_train + n_val]),
            ("test", images[n_train + n_val:]),
        )
        for split, items in splits:
            for p in items:
                dst = out_root / split / c / p.name
                symlink_or_copy(p, dst)

    return out_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isic_root", type=str, default=None)
    ap.add_argument("--odir_root", type=str, default=None)
    args = ap.parse_args()

    if args.isic_root:
        print("Preparing ISIC2018...", file=sys.stderr)
        isic_out = prepare_isic2018(Path(args.isic_root))
        print(f"ISIC2018 prepared at: {isic_out}")
    if args.odir_root:
        print("Preparing ODIR...", file=sys.stderr)
        odir_out = prepare_odir(Path(args.odir_root))
        print(f"ODIR prepared at: {odir_out}")


if __name__ == "__main__":
    main()
