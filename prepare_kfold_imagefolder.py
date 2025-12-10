#!/usr/bin/env python3
import argparse
import os
import random
from pathlib import Path


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        if dst.exists():
            return
        os.symlink(src, dst)
    except Exception:
        import shutil
        shutil.copy2(src, dst)


def list_images(d: Path):
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        imgs.extend(d.glob(ext))
    return sorted(imgs)


def main():
    ap = argparse.ArgumentParser(description="Prepare K-fold splits for imagefolder dataset")
    ap.add_argument("--root", required=True, help="Original imagefolder root (contains train/val/test)")
    ap.add_argument("--out", required=True, help="Output directory for k-fold splits")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    random.seed(args.seed)

    # Collect per-class images from train + val
    classes = sorted([p.name for p in (root / 'train').iterdir() if p.is_dir()])
    per_class = {}
    for c in classes:
        items = list_images(root / 'train' / c) + list_images(root / 'val' / c)
        random.shuffle(items)
        per_class[c] = items

    # Test set is kept as-is
    test_map = {}
    for c in classes:
        test_map[c] = list_images(root / 'test' / c)

    k = max(2, int(args.k))
    for fold in range(k):
        fold_root = out / f"fold_{fold}"
        for split in ("train", "val", "test"):
            for c in classes:
                ensure_dir(fold_root / split / c)
        # Build split
        for c, items in per_class.items():
            n = len(items)
            fold_size = max(1, n // k)
            start = fold * fold_size
            end = n if fold == k - 1 else min(n, start + fold_size)
            val_items = items[start:end]
            train_items = items[:start] + items[end:]
            for p in train_items:
                dst = fold_root / 'train' / c / p.name
                symlink_or_copy(p, dst)
            for p in val_items:
                dst = fold_root / 'val' / c / p.name
                symlink_or_copy(p, dst)
            for p in test_map[c]:
                dst = fold_root / 'test' / c / p.name
                symlink_or_copy(p, dst)

    print(f"Prepared {k}-fold splits under {out}")


if __name__ == '__main__':
    main()

