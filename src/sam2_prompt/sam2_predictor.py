"""
SAM 2 region-prompted defect segmentation.

For each candidate region from src.postprocess.regions, SAM 2 gets a prompt
built by one of four strategies (compared in Experiment D):

  "point"        one positive point at the region's peak (the original method)
  "multi_point"  n_pos positive points spread over the region's core
                 (farthest-point sampling, so elongated scratches get points
                 along their length rather than just at one spot)
  "box_point"    region bounding box (padded) + the peak point
  "box_pos_neg"  box + n_pos positives + n_neg negatives sampled from clearly
                 normal pixels around the region (default)

SAM returns 3 candidate masks (multimask_output=True, recommended by Meta for
ambiguous prompts). A candidate is ACCEPTED only if most of it lies inside the
dilated heatmap region ("containment" >= min_containment). This rejects the
typical failure where a point on a scratch makes SAM return the whole screen
or a finger. Among accepted masks the highest SAM IoU-prediction wins. If
none is accepted, the heatmap region itself is used (fallback="region") and
the fallback is recorded so it can be reported.

Containment rather than IoU is deliberate: a correct thin scratch mask is much
smaller than the blob-shaped heatmap region, so IoU would penalize the right
answer, while containment only penalizes leaking outside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from src.heatmap.heatmap import Peak
from src.postprocess.regions import Region

PROMPT_STRATEGIES = ("point", "multi_point", "box_point", "box_pos_neg")


@dataclass
class RegionSegmentation:
    mask: np.ndarray
    accepted: bool
    used_fallback: bool
    sam_score: float | None
    containment: float | None
    points: np.ndarray
    labels: np.ndarray
    box: np.ndarray | None


@dataclass
class SegmentationResult:
    mask: np.ndarray                      # (H, W) uint8 union of per-region masks
    per_region: list[RegionSegmentation] = field(default_factory=list)
    per_peak_scores: list[float] = field(default_factory=list)  # legacy field

    @property
    def n_fallbacks(self) -> int:
        return sum(r.used_fallback for r in self.per_region)


# ---------------------------------------------------------------- prompt building
def farthest_point_sampling(coords: np.ndarray, n: int, start: int = 0) -> np.ndarray:
    """Indices of n well-spread points from coords (N, 2)."""
    n = min(n, len(coords))
    if n <= 0:
        return np.empty(0, dtype=int)
    chosen = [start]
    d = np.linalg.norm(coords - coords[start], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(d))
        if d[nxt] <= 0:
            break
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(coords - coords[nxt], axis=1))
    return np.asarray(chosen)


def _subsample(coords: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if len(coords) > max_n:
        coords = coords[rng.choice(len(coords), max_n, replace=False)]
    return coords


def padded_box(bbox: tuple[int, int, int, int], shape: tuple[int, int], pad_frac: float, min_pad: int = 4) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    h, w = shape
    px = max(min_pad, int(round((x1 - x0) * pad_frac)))
    py = max(min_pad, int(round((y1 - y0) * pad_frac)))
    return np.array([max(0, x0 - px), max(0, y0 - py), min(w - 1, x1 + px), min(h - 1, y1 + py)], dtype=np.float32)


def build_prompts(
    region: Region,
    heatmap: np.ndarray,
    low_threshold: float,
    strategy: str = "box_pos_neg",
    n_pos: int = 3,
    n_neg: int = 3,
    box_pad_frac: float = 0.15,
    valid_mask: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Returns (points (N,2) as x,y; labels (N,) 1=pos 0=neg; box [x0,y0,x1,y1] or None)."""
    if strategy not in PROMPT_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {PROMPT_STRATEGIES}")
    rng = np.random.default_rng(seed)
    peak = np.array([region.peak_xy], dtype=np.float32)
    box = padded_box(region.bbox, heatmap.shape, box_pad_frac) if strategy.startswith("box") else None

    if strategy in ("point", "box_point"):
        return peak, np.array([1]), box

    # Positive points: spread over the region's core (above-median scores).
    ys, xs = np.nonzero(region.mask)
    vals = heatmap[ys, xs]
    core = vals >= np.median(vals)
    pts = _subsample(np.stack([xs[core], ys[core]], 1).astype(np.float32), 4000, rng)
    start = int(np.argmin(np.linalg.norm(pts - peak[0], axis=1)))
    pos = pts[farthest_point_sampling(pts, n_pos, start)]
    if strategy == "multi_point":
        return pos, np.ones(len(pos), dtype=int), None

    # Negative points: clearly-normal pixels in a ring around the region.
    from scipy.ndimage import binary_dilation

    ring_box = padded_box(region.bbox, heatmap.shape, box_pad_frac * 2 + 0.1, min_pad=8).astype(int)
    x0, y0, x1, y1 = ring_box
    zone = np.zeros(heatmap.shape, dtype=bool)
    zone[y0 : y1 + 1, x0 : x1 + 1] = True
    grown = binary_dilation(region.mask, iterations=3)
    cand = zone & ~grown & (heatmap < low_threshold)
    if valid_mask is not None:
        cand &= valid_mask
    nys, nxs = np.nonzero(cand)
    if len(nxs) and n_neg > 0:
        npts = _subsample(np.stack([nxs, nys], 1).astype(np.float32), 4000, rng)
        neg = npts[farthest_point_sampling(npts, n_neg, int(rng.integers(len(npts))))]
    else:
        neg = np.empty((0, 2), np.float32)
    points = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
    return points, labels, box


def select_mask(
    masks: np.ndarray, scores: np.ndarray, region_dilated: np.ndarray, min_containment: float, min_area: int = 5
) -> tuple[int | None, list[float]]:
    """Index of the accepted mask with the highest SAM score (or None), plus containments."""
    containments, best, best_score = [], None, -np.inf
    for i, m in enumerate(masks):
        m = m.astype(bool)
        area = int(m.sum())
        c = float((m & region_dilated).sum() / area) if area else 0.0
        containments.append(c)
        if area >= min_area and c >= min_containment and scores[i] > best_score:
            best, best_score = i, float(scores[i])
    return best, containments


# ---------------------------------------------------------------- SAM 2 wrapper
def _build_sam2(model_cfg: str, checkpoint_path: str, device: str):
    """build_sam2 with config-name fallbacks (bare name vs configs/sam2.1/...)."""
    from sam2.build_sam import build_sam2

    candidates = [model_cfg]
    base = model_cfg.split("/")[-1]
    if base.startswith("sam2.1"):
        candidates.append(f"configs/sam2.1/{base}")
    elif base.startswith("sam2"):
        candidates.append(f"configs/sam2/{base}")
    last_err = None
    for cfg in dict.fromkeys(candidates):
        try:
            return build_sam2(cfg, checkpoint_path, device=device)
        except Exception as e:  # hydra raises MissingConfigException etc.
            last_err = e
    raise RuntimeError(f"Could not build SAM 2 with any of {candidates}: {last_err}")


class Sam2DefectSegmenter:
    """checkpoint_path / model_cfg: e.g. 'checkpoints/sam2.1_hiera_small.pt' and
    'configs/sam2.1/sam2.1_hiera_s.yaml' (bare 'sam2.1_hiera_s.yaml' also works).
    predictor: inject any object with set_image/predict (used by the tests)."""

    def __init__(self, checkpoint_path: str | None = None, model_cfg: str | None = None,
                 device: str | None = None, predictor=None):
        if predictor is not None:
            self.predictor = predictor
            self.device = device or "cpu"
            return
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SAM2ImagePredictor(_build_sam2(model_cfg, checkpoint_path, self.device))
        self._image_key = None

    # low-level access (also used by the screen-ROI detector)
    def set_image(self, image_np: np.ndarray) -> None:
        import hashlib

        key = (image_np.shape, hashlib.md5(np.ascontiguousarray(image_np[::8, ::8]).tobytes()).hexdigest())
        if getattr(self, "_image_key", None) != key:
            self.predictor.set_image(image_np)
            self._image_key = key

    def predict(self, points=None, labels=None, box=None, multimask_output=True) -> tuple[np.ndarray, np.ndarray]:
        kwargs = {"multimask_output": multimask_output}
        if points is not None and len(points):
            kwargs["point_coords"] = np.asarray(points, np.float32)
            kwargs["point_labels"] = np.asarray(labels, np.int32)
        if box is not None:
            kwargs["box"] = np.asarray(box, np.float32)
        masks, scores, _ = self.predictor.predict(**kwargs)
        return np.asarray(masks) > 0.5, np.asarray(scores, np.float32).reshape(-1)

    def segment_regions(
        self,
        image: Image.Image,
        regions: list[Region],
        heatmap: np.ndarray,
        low_threshold: float,
        strategy: str = "box_pos_neg",
        n_pos: int = 3,
        n_neg: int = 3,
        box_pad_frac: float = 0.15,
        min_containment: float = 0.6,
        containment_dilate_px: int = 8,
        valid_mask: np.ndarray | None = None,
        fallback: str = "region",
    ) -> SegmentationResult:
        from scipy.ndimage import binary_dilation

        image_np = np.asarray(image.convert("RGB"))
        if image_np.shape[:2] != heatmap.shape:
            raise ValueError(f"image {image_np.shape[:2]} and heatmap {heatmap.shape} must have the same size")
        combined = np.zeros(heatmap.shape, np.uint8)
        if not regions:
            return SegmentationResult(combined)
        self.set_image(image_np)

        results = []
        for i, region in enumerate(regions):
            pts, lbl, box = build_prompts(region, heatmap, low_threshold, strategy, n_pos, n_neg,
                                          box_pad_frac, valid_mask, seed=i)
            masks, scores = self.predict(pts, lbl, box, multimask_output=True)
            dil = binary_dilation(region.mask, iterations=max(1, int(containment_dilate_px)))
            if valid_mask is not None:
                masks = masks & valid_mask[None]
            best, conts = select_mask(masks, scores, dil, min_containment)
            if best is not None:
                m = masks[best]
                rs = RegionSegmentation(m, True, False, float(scores[best]), conts[best], pts, lbl, box)
            else:
                m = region.mask if fallback == "region" else np.zeros_like(region.mask)
                rs = RegionSegmentation(m, False, fallback == "region", None,
                                        max(conts) if conts else None, pts, lbl, box)
            combined[m] = 1
            results.append(rs)
        return SegmentationResult(combined, results, [r.sam_score or 0.0 for r in results])

    def segment(self, image: Image.Image, peaks: list[Peak], multimask_output: bool = False) -> SegmentationResult:
        """LEGACY: one positive point per peak, masks unioned with no filtering (baseline)."""
        image_np = np.asarray(image.convert("RGB"))
        combined = np.zeros(image_np.shape[:2], np.uint8)
        if not peaks:
            return SegmentationResult(combined)
        self.set_image(image_np)
        scores_out = []
        for p in peaks:
            masks, scores = self.predict(np.array([[p.x, p.y]]), np.array([1]), None, multimask_output)
            b = int(np.argmax(scores))
            combined |= masks[b].astype(np.uint8)
            scores_out.append(float(scores[b]))
        return SegmentationResult(combined, [], scores_out)
