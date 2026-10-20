"""Plotting helpers for the notebook and the report."""

from __future__ import annotations

import numpy as np

from src.heatmap.heatmap import normalize_for_display


def overlay(image, mask, color=(255, 40, 40), alpha=0.45) -> np.ndarray:
    img = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    m = mask.astype(bool)
    img[m] = (1 - alpha) * img[m] + alpha * np.array(color, np.float32)
    return img.astype(np.uint8)


def heatmap_display(result_or_heat, calibration=None):
    """Heatmap scaled with a FIXED range [p50 of normals .. image threshold] when a
    calibration is given, so colours mean the same thing in every image."""
    heat = getattr(result_or_heat, "heatmap", result_or_heat)
    cal = calibration or getattr(result_or_heat, "calibration", None)
    if cal is not None:
        lo = cal.details.get("normal_pixel_percentiles", {}).get("50", float(heat.min()))
        return normalize_for_display(heat, lo, cal.image_threshold * 1.25)
    return normalize_for_display(heat)


def show_inspection(result, gt=None, title=None, axes=None, show=True):
    import matplotlib.pyplot as plt

    n = 4 + (gt is not None)
    if axes is None:
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6))
    img = result.image
    axes[0].imshow(img)
    axes[0].set_title(title or ("DEFECTIVE" if result.is_defective else "NORMAL"))
    axes[1].imshow(heatmap_display(result), cmap="inferno", vmin=0, vmax=1)
    if not result.valid_mask.all():
        axes[1].contour(result.valid_mask, levels=[0.5], colors="cyan", linewidths=0.8)
    axes[1].set_title(f"raw heatmap (score {result.image_score:.3f} / thr {result.calibration.image_threshold:.3f})",
                      fontsize=8)
    axes[2].imshow(overlay(img, result.heatmap_mask, (255, 170, 0)))
    for r in result.regions:
        x0, y0, x1, y1 = r.bbox
        axes[2].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=1))
    axes[2].set_title(f"heatmap regions ({len(result.regions)})")
    axes[3].imshow(overlay(img, result.final_mask))
    seg = result.segmentation
    if seg is not None:
        for rs in seg.per_region:
            pos, neg = rs.points[rs.labels == 1], rs.points[rs.labels == 0]
            axes[3].scatter(pos[:, 0], pos[:, 1], c="lime", s=14, marker="o")
            if len(neg):
                axes[3].scatter(neg[:, 0], neg[:, 1], c="red", s=14, marker="x")
            if rs.box is not None:
                x0, y0, x1, y1 = rs.box
                axes[3].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=0.8))
        axes[3].set_title(f"SAM 2 mask ({seg.n_fallbacks} fallback)")
    else:
        axes[3].set_title("final mask (heatmap)")
    if gt is not None:
        axes[4].imshow(gt, cmap="gray")
        axes[4].set_title("ground truth")
    for a in axes:
        a.axis("off")
    if show and axes is not None:
        plt.tight_layout()
    return axes


def show_roi(photo, roi):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].imshow(photo)
    q = np.vstack([roi.quad, roi.quad[:1]])
    ax[0].plot(q[:, 0], q[:, 1], "c-", lw=2)
    ax[0].set_title(f"detected screen ({roi.method})")
    ax[1].imshow(roi.image)
    ax[1].set_title("warped screen")
    ax[2].imshow(roi.valid_mask, cmap="gray")
    ax[2].set_title("valid (scored) area")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
