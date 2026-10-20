"""
Evaluation metrics.

Continuous heatmaps (threshold-free):
  pixel_auroc  pooled pixel-level ROC-AUC
  pro_score    Per-Region Overlap AUC up to FPR 0.3, normalized to [0, 1]
               (MVTec AD convention: curve starts at (0, 0), is interpolated
               at exactly FPR=0.3, integrated with the trapezoid rule)
Both are histogram-based with adaptive (quantile) bin edges, so they run on
hundreds of millions of pixels without sorting everything.

Image level:
  image_auroc  normal vs defective images from one score per image

Binary masks (SAM 2 output or thresholded heatmaps) -- AUROC/PRO degenerate
on 0/1 maps, so use these instead:
  binary_segmentation_metrics  pooled & per-image IoU / Dice / precision /
               recall, GT-region recall, false-positive components
"""

from __future__ import annotations

import numpy as np


def _adaptive_edges(heatmaps: list[np.ndarray], n_bins: int, sample: int = 2_000_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = sum(h.size for h in heatmaps)
    per = max(1, int(sample * 1.0 / max(len(heatmaps), 1)))
    parts = []
    for h in heatmaps:
        f = h.reshape(-1)
        parts.append(f if f.size <= per else f[rng.choice(f.size, per, replace=False)])
    s = np.concatenate(parts)
    lo = min(float(h.min()) for h in heatmaps)
    hi = max(float(h.max()) for h in heatmaps)
    q = np.quantile(s, np.linspace(0.0, 1.0, n_bins + 1)) if total else np.array([lo, hi])
    lin = np.linspace(lo, hi, n_bins // 4 + 1)  # also cover sparse tails evenly
    edges = np.unique(np.concatenate([[lo], q, lin, [hi]]))
    return edges


def _bin(h: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, h.reshape(-1), side="right") - 1, 0, len(edges) - 1)


def _ge_counts(hist: np.ndarray) -> np.ndarray:
    """count of values >= edges[j], for each j."""
    return np.cumsum(hist[..., ::-1], axis=-1)[..., ::-1]


def pixel_auroc(heatmaps: list[np.ndarray], gt_masks: list[np.ndarray], n_bins: int = 20000, exact: bool | None = None) -> float:
    if len(heatmaps) != len(gt_masks):
        raise ValueError("heatmaps and gt_masks must be the same length")
    total = sum(h.size for h in heatmaps)
    n_pos = sum(int(g.astype(bool).sum()) for g in gt_masks)
    if n_pos == 0 or n_pos == total:
        raise ValueError("Pixel AUROC needs both defect and normal pixels in the evaluated set.")
    if exact is None:
        exact = total <= 5_000_000
    if exact:
        from sklearn.metrics import roc_auc_score

        y = np.concatenate([g.reshape(-1).astype(bool) for g in gt_masks])
        s = np.concatenate([h.reshape(-1).astype(np.float32) for h in heatmaps])
        return float(roc_auc_score(y, s))

    edges = _adaptive_edges(heatmaps, n_bins)
    pos = np.zeros(len(edges), np.int64)
    neg = np.zeros(len(edges), np.int64)
    for h, g in zip(heatmaps, gt_masks):
        b = _bin(h, edges)
        gm = g.reshape(-1).astype(bool)
        pos += np.bincount(b[gm], minlength=len(edges))
        neg += np.bincount(b[~gm], minlength=len(edges))
    # AUROC = P(score_pos > score_neg) + 0.5 P(equal bin)
    neg_below = np.cumsum(neg) - neg
    auc = (pos * neg_below).sum() + 0.5 * (pos * neg).sum()
    return float(auc / (pos.sum() * neg.sum()))


def pro_curve(heatmaps: list[np.ndarray], gt_masks: list[np.ndarray], n_bins: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, pro) arrays sorted by increasing FPR, starting at (0, 0)."""
    from skimage.measure import label as cc_label

    edges = _adaptive_edges(heatmaps, n_bins)
    neg = np.zeros(len(edges), np.int64)
    region_hists, region_sizes = [], []
    for h, g in zip(heatmaps, gt_masks):
        b = _bin(h, edges)
        gm = g.reshape(-1).astype(bool)
        neg += np.bincount(b[~gm], minlength=len(edges))
        lab = cc_label(g.astype(bool), connectivity=2).reshape(-1)
        if lab.max() == 0:
            continue
        idx = lab > 0
        # 2D histogram (region, bin) in one pass
        nr, nb = int(lab.max()), len(edges)
        rh = np.bincount((lab[idx] - 1).astype(np.int64) * nb + b[idx], minlength=nr * nb).reshape(nr, nb)
        region_hists.append(rh)
        region_sizes.append(rh.sum(1))
    if not region_hists:
        raise ValueError("PRO needs at least one ground-truth defect region.")
    rh = np.concatenate(region_hists)
    sizes = np.concatenate(region_sizes).astype(np.float64)
    pro = (_ge_counts(rh) / sizes[:, None]).mean(0)
    fpr = _ge_counts(neg) / max(neg.sum(), 1)
    order = np.argsort(fpr, kind="stable")
    fpr, pro = fpr[order], pro[order]
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], pro])


def pro_score(heatmaps: list[np.ndarray], gt_masks: list[np.ndarray], fpr_cap: float = 0.3, n_bins: int = 4000) -> float:
    fpr, pro = pro_curve(heatmaps, gt_masks, n_bins)
    # keep the running max of PRO for equal FPR (monotone curve), then cut at fpr_cap
    pro_at_cap = float(np.interp(fpr_cap, fpr, pro))
    keep = fpr <= fpr_cap
    x = np.concatenate([fpr[keep], [fpr_cap]])
    y = np.concatenate([pro[keep], [pro_at_cap]])
    trap = getattr(np, "trapezoid", None) or np.trapz  # np.trapz is deprecated in NumPy 2
    return float(np.clip(trap(y, x) / fpr_cap, 0.0, 1.0))


def evaluate(heatmaps: list[np.ndarray], gt_masks: list[np.ndarray]) -> dict[str, float]:
    """Threshold-free heatmap metrics (same interface as the original)."""
    return {"pixel_auroc": pixel_auroc(heatmaps, gt_masks), "pro_score": pro_score(heatmaps, gt_masks)}


def image_auroc(scores: list[float] | np.ndarray, labels: list[int] | np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(labels).astype(int), np.asarray(scores, np.float64)))


def image_decision_metrics(scores, labels, threshold: float) -> dict:
    s, y = np.asarray(scores, np.float64), np.asarray(labels).astype(bool)
    pred = s > threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    return {
        "threshold": float(threshold),
        "tpr_defect_detected": tp / max(tp + fn, 1),
        "fpr_normal_flagged": fp / max(fp + tn, 1),
        "accuracy": (tp + tn) / max(len(s), 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def binary_segmentation_metrics(
    pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], region_hit_frac: float = 0.1
) -> dict:
    from skimage.measure import label as cc_label

    tp = fp = fn = 0
    ious, dices, precs, recs = [], [], [], []
    n_gt_regions = n_gt_hit = n_fp_components = 0
    for p, g in zip(pred_masks, gt_masks):
        p, g = p.astype(bool), g.astype(bool)
        itp, ifp, ifn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
        tp, fp, fn = tp + itp, fp + ifp, fn + ifn
        if g.any() or p.any():
            ious.append(itp / max(itp + ifp + ifn, 1))
            dices.append(2 * itp / max(2 * itp + ifp + ifn, 1))
        else:
            ious.append(1.0)
            dices.append(1.0)
        precs.append(itp / max(itp + ifp, 1) if p.any() else float(not g.any()))
        recs.append(itp / max(itp + ifn, 1) if g.any() else 1.0)

        gl = cc_label(g, connectivity=2)
        for r in range(1, gl.max() + 1):
            rm = gl == r
            n_gt_regions += 1
            n_gt_hit += int(p[rm].mean() >= region_hit_frac)
        pl = cc_label(p, connectivity=2)
        for r in range(1, pl.max() + 1):
            n_fp_components += int(not g[pl == r].any())

    return {
        "pooled_iou": tp / max(tp + fp + fn, 1),
        "pooled_dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "pooled_precision": tp / max(tp + fp, 1),
        "pooled_recall": tp / max(tp + fn, 1),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "mean_precision": float(np.mean(precs)) if precs else 0.0,
        "mean_recall": float(np.mean(recs)) if recs else 0.0,
        "gt_region_recall": n_gt_hit / max(n_gt_regions, 1),
        "false_positive_components_per_image": n_fp_components / max(len(pred_masks), 1),
        "n_images": len(pred_masks),
    }
