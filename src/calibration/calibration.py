"""
Threshold calibration: turning raw anomaly scores into decisions.

Two thresholds, both in raw cosine-distance units:

  image_threshold (HIGH): an image is defective iff some valid pixel of its
      heatmap exceeds this. Label-free default: the largest heatmap value seen
      on any normal reference image under leave-one-out scoring, i.e. "more
      anomalous than anything we've seen on a clean screen".

  pixel_threshold (LOW): once a defect is detected, its region grows out to
      pixels above this (hysteresis thresholding, see src.postprocess.regions).
      Label-free default: a high percentile of leave-one-out normal pixel scores.

Leave-one-out matters: scoring a reference image against a bank that contains
its own patches gives distance ~0 everywhere and a useless threshold.
Caveat to state in the report: LOO scores use a bank with one image fewer,
so they are slightly pessimistic (thresholds slightly conservative).

Label-using alternatives (for comparison / upper bounds, clearly labelled so):
  * calibrate_pixel_from_background: percentile of val defect images'
    background pixels (uses GT masks to find background).
  * tune_pixel_threshold_on_val: threshold maximizing pooled Dice on val.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from src.feature_extraction.dinov2_extractor import PatchGrid
from src.heatmap.heatmap import upsample_heatmap
from src.memory_bank.memory_bank import MemoryBank


@dataclass
class Calibration:
    pixel_threshold: float
    image_threshold: float
    method: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(**d)

    def with_pixel_threshold(self, t: float, method_suffix: str) -> "Calibration":
        t = float(min(t, self.image_threshold))
        return Calibration(t, self.image_threshold, f"{self.method}+{method_suffix}", dict(self.details))


def loo_normal_heatmaps(
    bank: MemoryBank,
    normal_grids: list[PatchGrid],
    out_sizes: list[tuple[int, int]],
    smooth_patches: float = 0.5,
    group_ids: list[int] | None = None,
    patch_ok: list[np.ndarray | None] | None = None,
) -> list[np.ndarray]:
    """Leave-one-out heatmaps for the bank's own reference images.

    normal_grids must be the grids the bank was built from; group_ids must match
    what was passed to bank.build (defaults to 0..n-1).
    """
    group_ids = list(range(len(normal_grids))) if group_ids is None else list(group_ids)
    if len(set(group_ids)) < 3:
        raise ValueError("Leave-one-out calibration needs at least 3 distinct reference images.")
    from src.heatmap.heatmap import neutralize_invalid

    patch_ok = patch_ok or [None] * len(normal_grids)
    return [
        upsample_heatmap(neutralize_invalid(bank.score_grid(g, exclude_image=gid), po), size, smooth_patches)
        for g, gid, size, po in zip(normal_grids, group_ids, out_sizes, patch_ok)
    ]


def _valid_values(h: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    return h[valid] if valid is not None else h.reshape(-1)


def calibrate_from_normals(
    normal_heatmaps: list[np.ndarray],
    valid_masks: list[np.ndarray | None] | None = None,
    pixel_percentile: float = 99.5,
    image_rule: str = "max",
    image_margin: float = 0.0,
    image_k_std: float = 3.0,
) -> Calibration:
    valid_masks = valid_masks or [None] * len(normal_heatmaps)
    vals = [_valid_values(h, v) for h, v in zip(normal_heatmaps, valid_masks)]
    pix = np.concatenate(vals)
    img_scores = np.array([v.max() for v in vals], dtype=np.float64)

    pixel_t = float(np.percentile(pix, pixel_percentile))
    if image_rule == "max":
        image_t = float(img_scores.max() * (1.0 + image_margin))
    elif image_rule == "mean_std":
        image_t = float(img_scores.mean() + image_k_std * img_scores.std())
    else:
        raise ValueError(f"Unknown image_rule {image_rule!r}")
    image_t = max(image_t, pixel_t)

    return Calibration(
        pixel_threshold=pixel_t,
        image_threshold=image_t,
        method=f"loo_normals(p{pixel_percentile},{image_rule})",
        details={
            "n_normal_images": len(normal_heatmaps),
            "normal_image_scores": img_scores.round(6).tolist(),
            "normal_pixel_percentiles": {
                str(p): float(np.percentile(pix, p)) for p in (50, 90, 99, 99.5, 99.9)
            },
        },
    )


def background_mask(gt: np.ndarray, dilate_px: int = 5) -> np.ndarray:
    """True where a pixel is safely background (GT defect dilated then inverted)."""
    from scipy.ndimage import binary_dilation

    g = gt.astype(bool)
    if dilate_px > 0:
        g = binary_dilation(g, iterations=int(dilate_px))
    return ~g


def calibrate_pixel_from_background(
    heatmaps: list[np.ndarray], gt_masks: list[np.ndarray], percentile: float = 99.5, dilate_px: int = 5
) -> float:
    """LABEL-USING: percentile of background-pixel scores on (val) defect images."""
    vals = np.concatenate([h[background_mask(g, dilate_px)] for h, g in zip(heatmaps, gt_masks)])
    return float(np.percentile(vals, percentile))


def tune_pixel_threshold_on_val(
    heatmaps: list[np.ndarray], gt_masks: list[np.ndarray], n_bins: int = 1000
) -> tuple[float, float]:
    """LABEL-USING upper bound: threshold maximizing pooled Dice on the given (val) set.
    Returns (threshold, best_dice)."""
    scores = np.concatenate([h.reshape(-1) for h in heatmaps]).astype(np.float32)
    labels = np.concatenate([g.reshape(-1) for g in gt_masks]).astype(bool)
    edges = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, len(edges) - 1)
    pos = np.bincount(idx[labels], minlength=len(edges))
    neg = np.bincount(idx[~labels], minlength=len(edges))
    tp = np.cumsum(pos[::-1])[::-1]   # predicted positive = score >= edges[j]
    fp = np.cumsum(neg[::-1])[::-1]
    fn = labels.sum() - tp
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    j = int(np.argmax(dice))
    return float(edges[j]), float(dice[j])


def score_distribution_diagnostic(
    normal_heatmaps: list[np.ndarray],
    defect_heatmaps: list[np.ndarray],
    gt_masks: list[np.ndarray],
    calibration: Calibration,
    dilate_px: int | None = None,
) -> dict:
    """Compares score distributions of (a) LOO normal pixels, (b) background pixels of
    defect images, (c) defect pixels. If (b) sits well above (a), the normal reference
    images differ systematically from the defect images (e.g. PNG vs JPEG, capture
    session) and label-free thresholds will over-segment. Try normal_jpeg_quality."""
    # "Background" must be FAR from any defect: heatmap blobs legitimately extend a patch or two
    # beyond the true defect, and that blur is not domain shift. Default: 3% of the long side
    # (~2 patch footprints at the default settings).
    def dil(g):
        return int(round(0.03 * max(g.shape))) if dilate_px is None else int(dilate_px)

    norm = np.concatenate([h.reshape(-1) for h in normal_heatmaps])
    bg = np.concatenate([h[background_mask(g, dil(g))] for h, g in zip(defect_heatmaps, gt_masks)])
    fg = np.concatenate([h[g.astype(bool)] for h, g in zip(defect_heatmaps, gt_masks)])
    pct = (50, 90, 99, 99.5, 99.9)
    t = calibration.pixel_threshold
    d = {
        "normal_loo_pixels": {str(p): float(np.percentile(norm, p)) for p in pct},
        "defect_image_background": {str(p): float(np.percentile(bg, p)) for p in pct},
        "defect_pixels": {str(p): float(np.percentile(fg, p)) for p in pct} if fg.size else {},
        "frac_background_above_pixel_threshold": float((bg > t).mean()),
        "frac_normal_above_pixel_threshold": float((norm > t).mean()),
        "frac_defect_above_pixel_threshold": float((fg > t).mean()) if fg.size else None,
        "background_images_flagged": float(
            np.mean([h[background_mask(g, dil(g))].max() > calibration.image_threshold
                     for h, g in zip(defect_heatmaps, gt_masks)])
        ),
    }
    ratio = d["frac_background_above_pixel_threshold"] / max(d["frac_normal_above_pixel_threshold"], 1e-9)
    d["background_vs_normal_exceedance_ratio"] = float(ratio)
    # Scale-free shift test, robust to heatmap blur around defects: where does each defect image's
    # MEDIAN background score fall within the normal LOO distribution? ~50 = no shift, ~100 = shift.
    norm_sorted = np.sort(norm)
    med_pct = [100.0 * np.searchsorted(norm_sorted, np.median(h[background_mask(g, dil(g))])) / norm_sorted.size
               for h, g in zip(defect_heatmaps, gt_masks)]
    d["background_median_percentile_in_normals"] = float(np.median(med_pct))
    d["warning"] = (
        "Typical background pixels of the defect images score higher than ~90% of normal pixels: the "
        "reference images differ systematically from the test images (JPEG, capture session, lighting). "
        "Label-free thresholds will over-segment. Check normal_jpeg_quality / use background calibration."
        if d["background_median_percentile_in_normals"] > 90
        else None
    )
    return d
