"""
Anomaly heatmap generation.

Key change: heatmaps now stay in *raw cosine-distance units*. The old
pipeline min-max normalized every heatmap per image and then thresholded at
0.6. Because every normalized map has a maximum of exactly 1.0, that
threshold was always exceeded, so every image (even a clean one) was
reported as defective and SAM 2 was always prompted somewhere.

Raw scores are comparable across images, so a single threshold calibrated
on normal images (src/calibration) can say "this image is normal".
`normalize_for_display` exists only for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


def work_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Aspect-preserving (width, height) with the given long side."""
    scale = long_side / max(width, height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def upsample_heatmap(
    score_grid: np.ndarray,
    output_size: tuple[int, int],
    smooth_patches: float = 0.5,
    smooth_sigma: float | None = None,
) -> np.ndarray:
    """Bilinear-upsample a (grid_h, grid_w) score grid to output_size=(W, H), then Gaussian-smooth.

    smooth_patches: sigma expressed in patch footprints, so smoothing behaves the
        same at any resolution (the old fixed sigma=4 px was a no-op at 1920x1080,
        where one patch spans ~50 px). Ignored if smooth_sigma (pixels) is given.
    """
    img = Image.fromarray(np.asarray(score_grid, np.float32), mode="F")
    heat = np.asarray(img.resize(tuple(output_size), resample=Image.BILINEAR), dtype=np.float32)

    if smooth_sigma is None:
        gh, gw = score_grid.shape
        px_per_patch = 0.5 * (output_size[0] / gw + output_size[1] / gh)
        smooth_sigma = smooth_patches * px_per_patch
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter

        heat = gaussian_filter(heat, sigma=float(smooth_sigma)).astype(np.float32)
    return heat


def neutralize_invalid(score_grid: np.ndarray, patch_ok: np.ndarray | None) -> np.ndarray:
    """Replace scores of invalid patches (finger, bezel, off-screen) with the median valid score
    BEFORE upsampling/smoothing, so an occluder's very high score cannot bleed into valid pixels."""
    if patch_ok is None or patch_ok.all():
        return score_grid
    out = score_grid.astype(np.float32, copy=True)
    out[~patch_ok] = float(np.median(score_grid[patch_ok])) if patch_ok.any() else float(np.median(score_grid))
    return out


def px_per_patch(score_grid_shape: tuple[int, int], output_size: tuple[int, int]) -> float:
    gh, gw = score_grid_shape
    return 0.5 * (output_size[0] / gw + output_size[1] / gh)


def normalize_for_display(scores: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Min-max scale to [0, 1] FOR PLOTTING ONLY. Pass the calibrated thresholds as
    vmin/vmax to get a colour scale that is consistent across images."""
    lo = float(scores.min()) if vmin is None else float(vmin)
    hi = float(scores.max()) if vmax is None else float(vmax)
    if hi - lo < 1e-8:
        return np.zeros_like(scores, dtype=np.float32)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# Deprecated name kept so old notebook cells still run. Do NOT use its output for
# decisions or metrics: per-image normalization destroys cross-image comparability.
normalize_scores = normalize_for_display


@dataclass
class Peak:
    x: int
    y: int
    score: float


def find_peaks(heatmap: np.ndarray, threshold: float = 0.6, min_distance: int = 20, max_peaks: int = 10) -> list[Peak]:
    """LEGACY prompt generator (local maxima above a threshold), kept to reproduce the
    baseline in Experiment C. The improved pipeline uses src.postprocess.regions."""
    from skimage.feature import peak_local_max

    coords = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=threshold, num_peaks=max_peaks)
    peaks = [Peak(x=int(c[1]), y=int(c[0]), score=float(heatmap[c[0], c[1]])) for c in coords]
    return sorted(peaks, key=lambda p: p.score, reverse=True)
