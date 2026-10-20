"""
Dataset loader for the MSD (Mobile Phone Screen Surface Defect Segmentation) dataset.

Handles the as-downloaded Kaggle folder layout, which looks like:

    data/raw/MSD/archive (2)/
        good/               20 defect-free reference images  (0001.png ... 0020.png)
        oil/                400 defect images                (Oil_0001.jpg ...)
        scratch/            400 defect images                (Scr_0001.jpg ...)
        stain/              400 defect images                (Sta_0001.jpg ...)
        ground_truth_1/     800 palette masks for scratch + stain (Scr_*.png / Sta_*.png)
        ground_truth_2/     400 palette masks for oil            (Oil_*.png)

Ground-truth masks are palette ("P" mode) PNGs where 0 = background and any
non-zero index = a defect pixel. The exact non-zero index differs per class
(observed: scratch -> 3, stain -> 2, oil -> {1, 2}), so this loader always
binarizes masks as (index != 0) rather than relying on a specific label
value, since the DINOv2 + SAM 2 pipeline is class-agnostic (it only cares
about "defect" vs "normal").

This module has no third-party dependencies beyond Pillow and numpy so it can
be imported both in the lightweight sandbox (for inspection/tests) and in the
full Colab/GPU environment used to actually run the pipeline.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DEFECT_CLASSES = ("oil", "scratch", "stain")

# Kaggle download unzips to a folder literally named "archive (2)" by default.
# We resolve it defensively instead of hard-coding the parenthesized name.
_RAW_SUBDIR_CANDIDATES = ("archive (2)", "archive", ".")

_CLASS_TO_PREFIX = {"oil": "Oil", "scratch": "Scr", "stain": "Sta"}
_CLASS_TO_GT_DIR = {"oil": "ground_truth_2", "scratch": "ground_truth_1", "stain": "ground_truth_1"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path | None  # None for normal/reference images
    label: str  # "good", "oil", "scratch", or "stain"


def _resolve_root(msd_root: Path) -> Path:
    """Find the actual folder containing good/oil/scratch/stain/ground_truth_*."""
    for candidate in _RAW_SUBDIR_CANDIDATES:
        p = msd_root / candidate
        if (p / "good").is_dir():
            return p
    for p in sorted(msd_root.glob("*/")):  # any single nested folder, e.g. "archive (3)"
        if (p / "good").is_dir():
            return p
    raise FileNotFoundError(
        f"Could not locate MSD dataset contents under {msd_root}. "
        f"Expected a 'good' subfolder somewhere under it."
    )


_STD_LUMA_Q = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def estimate_jpeg_quality(path: str | Path) -> int | None:
    """Estimates the libjpeg quality factor of a JPEG from its luminance quantization
    table (label-free). Returns None for non-JPEG files or non-standard tables."""
    with Image.open(path) as im:
        q = getattr(im, "quantization", None)
    if not q or 0 not in q:
        return None
    table = np.sort(np.asarray(q[0], dtype=np.float64))
    std = np.sort(_STD_LUMA_Q)
    best, best_err = None, np.inf
    for quality in range(1, 101):  # brute force over libjpeg's scaling rule (order-independent)
        scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
        cand = np.clip(np.floor((std * scale + 50.0) / 100.0), 1, 255)
        err = np.abs(np.sort(cand) - table).mean()
        if err < best_err:
            best, best_err = quality, err
    return int(best) if best_err < 1.0 else None


class MSDDataset:
    """Convenience accessor over the MSD dataset folder."""

    def __init__(self, msd_root: str | Path):
        self.root = _resolve_root(Path(msd_root))
        self._validate()

    def _validate(self) -> None:
        required = ["good", "oil", "scratch", "stain", "ground_truth_1", "ground_truth_2"]
        missing = [d for d in required if not (self.root / d).is_dir()]
        if missing:
            raise FileNotFoundError(f"MSD dataset at {self.root} is missing folders: {missing}")

    def normal_samples(self) -> list[Sample]:
        """The 20 defect-free reference images used to build the DINOv2 memory bank."""
        good_dir = self.root / "good"
        return [Sample(p, None, "good") for p in sorted(good_dir.glob("*.png"))]

    def defect_samples(self, defect_class: str | None = None) -> list[Sample]:
        """All defect images (optionally filtered to one class), paired with their masks."""
        classes = [defect_class] if defect_class else list(DEFECT_CLASSES)
        samples: list[Sample] = []
        for cls in classes:
            if cls not in DEFECT_CLASSES:
                raise ValueError(f"Unknown defect class '{cls}', expected one of {DEFECT_CLASSES}")
            img_dir = self.root / cls
            gt_dir = self.root / _CLASS_TO_GT_DIR[cls]
            prefix = _CLASS_TO_PREFIX[cls]
            for img_path in sorted(img_dir.glob(f"{prefix}_*.jpg")):
                mask_path = gt_dir / f"{img_path.stem}.png"
                if not mask_path.exists():
                    raise FileNotFoundError(f"Missing mask for {img_path} at {mask_path}")
                samples.append(Sample(img_path, mask_path, cls))
        return samples

    def split(
        self, defect_class: str | None = None, val_frac: float = 0.2, test_frac: float = 0.2, seed: int = 42
    ) -> dict[str, list[Sample]]:
        """Reproducible, class-stratified 60/20/20 train/val/test split.

        Each class is shuffled with its own derived seed, and the all-class split
        (defect_class=None) is the *union* of the per-class splits. So the test
        set used for per-class numbers and for all-class numbers is the same set
        of images. (The previous version shuffled all 1,200 images together for
        defect_class=None, which produced a different test set.)

        Note: this is a seeded random split made for this project, not the
        official FDSNet/MSD split.
        """
        classes = [defect_class] if defect_class else list(DEFECT_CLASSES)
        out: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
        for cls in classes:
            samples = self.defect_samples(cls)
            rng = random.Random(seed + 1000 * DEFECT_CLASSES.index(cls))
            shuffled = samples[:]
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_test = int(n * test_frac)
            n_val = int(n * val_frac)
            out["test"] += shuffled[:n_test]
            out["val"] += shuffled[n_test : n_test + n_val]
            out["train"] += shuffled[n_test + n_val :]
        return out

    @staticmethod
    def load_image(sample: Sample, jpeg_quality: int | None = None) -> Image.Image:
        """Loads an RGB image.

        jpeg_quality: if set and the file is not already a JPEG, round-trips the
        image through JPEG at this quality. The 20 normal images are PNG while
        all defect images are JPEG, so without this the memory bank never sees
        JPEG compression artifacts. Phase 1's score diagnostic tells you whether
        that mismatch matters for your data.
        """
        img = Image.open(sample.image_path).convert("RGB")
        if jpeg_quality is not None and sample.image_path.suffix.lower() not in (".jpg", ".jpeg"):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=int(jpeg_quality))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img

    def estimate_defect_jpeg_quality(self, n: int = 12) -> int | None:
        """Median estimated JPEG quality over a few defect images (used to round-trip the PNG
        normals so the memory bank sees the same compression artifacts as the test images)."""
        qs = [estimate_jpeg_quality(s.image_path) for s in self.defect_samples()[:: max(1, 1200 // n)][:n]]
        qs = [q for q in qs if q is not None]
        return int(np.median(qs)) if qs else None

    @staticmethod
    def load_binary_mask(sample: Sample) -> np.ndarray:
        """Returns an (H, W) uint8 array: 1 = defect pixel, 0 = background."""
        if sample.mask_path is None:
            raise ValueError(f"Sample {sample.image_path} has no ground-truth mask (it is a normal image).")
        mask = np.array(Image.open(sample.mask_path))
        return (mask != 0).astype(np.uint8)


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw/MSD"
    ds = MSDDataset(root)
    normals = ds.normal_samples()
    defects = ds.defect_samples()
    print(f"Resolved dataset root: {ds.root}")
    print(f"Normal reference images: {len(normals)}")
    for cls in DEFECT_CLASSES:
        print(f"  {cls}: {len(ds.defect_samples(cls))} images")
    print(f"Total defect images: {len(defects)}")
    split = ds.split()
    print(f"Split -> train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
