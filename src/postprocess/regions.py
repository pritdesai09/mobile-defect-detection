"""
Heatmap -> candidate defect regions.

Replaces "any local maximum above 0.6 is a defect" with hysteresis
thresholding on raw scores:

  * a region must contain at least one pixel above image_threshold (HIGH),
    which ties localization to the image-level decision: no detection means
    no regions and no SAM 2 prompts;
  * it then grows through connected pixels above pixel_threshold (LOW);
  * pixels outside the valid mask (screen ROI, finger occlusion, bezel
    margin) can never seed or join a region;
  * components smaller than min_area are dropped as noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Region:
    mask: np.ndarray                 # (H, W) bool
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (x1/y1 exclusive)
    area: int
    peak_xy: tuple[int, int]
    peak_score: float
    mean_score: float


def border_valid_mask(shape: tuple[int, int], margin_frac: float) -> np.ndarray:
    """(H, W) bool, False within margin_frac of each border."""
    h, w = shape
    m = np.ones((h, w), dtype=bool)
    if margin_frac > 0:
        mx, my = int(round(w * margin_frac)), int(round(h * margin_frac))
        if my:
            m[:my] = m[-my:] = False
        if mx:
            m[:, :mx] = m[:, -mx:] = False
    return m


def resize_mask(mask: np.ndarray, size: tuple[int, int], mode: str = "any") -> np.ndarray:
    """Resize a binary mask to size=(W, H).

    mode="any": a target pixel is True if any covered source pixel is True
        (area interpolation > 0). Use for ground-truth masks so thin scratches
        survive downsampling.
    mode="majority": True if >= half the covered source pixels are True. Use
        for validity masks.
    """
    import cv2

    src = mask.astype(np.float32)
    if (src.shape[1], src.shape[0]) == tuple(size):
        return mask.astype(bool)
    shrinking = size[0] < src.shape[1]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    r = cv2.resize(src, tuple(size), interpolation=interp)
    return (r > 1e-6) if mode == "any" else (r >= 0.5)


def patch_valid_from_pixel_mask(valid: np.ndarray, grid_shape: tuple[int, int], min_frac: float = 0.5) -> np.ndarray:
    """Pixel validity mask -> (grid_h, grid_w) bool: patch is valid if >= min_frac of it is."""
    import cv2

    gh, gw = grid_shape
    frac = cv2.resize(valid.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    return frac >= min_frac


def occlusion_safe_mask(valid: np.ndarray, band_px: int) -> np.ndarray:
    """Shrinks the valid area by band_px around INVALID pixels only (image borders are not
    eroded). DINOv2 patches straddling a finger/bezel edge carry its content, and heatmap
    smoothing spreads it further, so pixels near an occlusion are not trustworthy."""
    if valid.all() or band_px <= 0:
        return valid
    from scipy.ndimage import binary_erosion

    return binary_erosion(valid, iterations=int(band_px), border_value=1)


def patch_valid_strict(valid: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """Patches safe to put in a memory bank: fully valid AND not adjacent to an invalid patch."""
    from scipy.ndimage import binary_erosion

    pv = patch_valid_from_pixel_mask(valid, grid_shape, min_frac=0.99)
    return pv if pv.all() else binary_erosion(pv, iterations=1, border_value=1)


def extract_regions(
    heatmap: np.ndarray,
    low: float,
    high: float,
    valid_mask: np.ndarray | None = None,
    min_area: int = 0,
    max_regions: int = 10,
) -> tuple[list[Region], np.ndarray]:
    """Returns (regions sorted by peak score, union mask as uint8)."""
    from skimage.filters import apply_hysteresis_threshold
    from skimage.measure import label, regionprops

    low = min(low, high)
    h = heatmap.astype(np.float32, copy=True)
    if valid_mask is not None:
        h[~valid_mask] = min(float(h.min()), low) - 1.0
    seeds_exist = bool((h > high).any())
    if not seeds_exist:
        return [], np.zeros(heatmap.shape, np.uint8)

    hyst = apply_hysteresis_threshold(h, low, high)
    lab = label(hyst, connectivity=2)
    regions: list[Region] = []
    for rp in regionprops(lab, intensity_image=heatmap):
        if rp.area < min_area:
            continue
        m = lab == rp.label
        vals = heatmap[m]
        iy, ix = np.unravel_index(np.argmax(np.where(m, heatmap, -np.inf)), heatmap.shape)
        y0, x0, y1, x1 = rp.bbox
        regions.append(
            Region(m, (int(x0), int(y0), int(x1), int(y1)), int(rp.area), (int(ix), int(iy)),
                   float(vals.max()), float(vals.mean()))
        )
    regions.sort(key=lambda r: r.peak_score, reverse=True)
    regions = regions[:max_regions]
    union = np.zeros(heatmap.shape, np.uint8)
    for r in regions:
        union[r.mask] = 1
    return regions, union
