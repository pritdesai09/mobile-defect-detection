"""
Evaluation runner + the project's experiments.

DINOv2 features for each evaluation image are extracted ONCE into an EvalSet
and reused across experiments (different k, bank subsets, thresholds and
prompt strategies only change the cheap scoring/post-processing steps).

  run_heatmap_eval        pixel AUROC / PRO on RAW heatmaps, per class, plus the
                          old per-image-normalized variant for comparison (issue 2)
  run_image_level_eval    normal-vs-defective image AUROC + decision rates (issue 1/2)
  exp_A_roi               live photos: full image vs manual crop vs auto ROI
  exp_B_thresholds        pixel-threshold sweep on val (label-free vs val-tuned)
  exp_C_peaks_vs_regions  legacy peaks@0.6 vs calibrated hysteresis regions
  exp_D_prompt_strategies SAM 2 prompt strategies vs thresholded heatmap
  exp_E_bank_size         reference-set size (with truly held-out normals)
  exp_F_final_masks       heatmap vs thresholded heatmap vs SAM 2 masks (+ legacy)
  exp_G_feature_ablation  (driven from the notebook: one call per feature config, on val)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from src.calibration.calibration import (
    Calibration,
    calibrate_from_normals,
    calibrate_pixel_from_background,
    loo_normal_heatmaps,
    tune_pixel_threshold_on_val,
)
from src.config import PipelineConfig
from src.evaluation.metrics import (
    binary_segmentation_metrics,
    evaluate,
    image_auroc,
    image_decision_metrics,
)
from src.heatmap.heatmap import find_peaks, normalize_for_display, px_per_patch, upsample_heatmap, work_size
from src.memory_bank.memory_bank import MemoryBank
from src.postprocess.regions import extract_regions, resize_mask


# ------------------------------------------------------------------ I/O helpers
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def save_json(obj, path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2)


def _progress(it, desc="", enable=True):
    if not enable:
        return it
    try:
        from tqdm.auto import tqdm

        return tqdm(it, desc=desc)
    except ImportError:
        return it


# ------------------------------------------------------------------ eval set
@dataclass
class EvalSet:
    name: str
    samples: list
    grids: list
    sizes: list
    gts: list
    labels: list = field(default_factory=list)

    def __len__(self):
        return len(self.samples)

    def subset(self, idx) -> "EvalSet":
        idx = list(idx)
        return EvalSet(self.name, [self.samples[i] for i in idx], [self.grids[i] for i in idx],
                       [self.sizes[i] for i in idx], [self.gts[i] for i in idx], [self.labels[i] for i in idx])


def build_eval_set(extractor, dataset, samples, cfg: PipelineConfig, name: str = "test",
                   batch: int = 8, progress: bool = True) -> EvalSet:
    """Extracts DINOv2 grids (stored float16) and loads GT masks at working resolution."""
    grids, sizes, gts = [], [], []
    for s in _progress(range(0, len(samples), batch), f"features[{name}]", progress):
        chunk = samples[s : s + batch]
        imgs = [dataset.load_image(x) for x in chunk]
        for img, g, smp in zip(imgs, extractor.extract_batch(imgs), chunk):
            size = work_size(img.width, img.height, cfg.work_long_side)
            grids.append(g.astype(np.float16))
            sizes.append(size)
            gts.append(resize_mask(dataset.load_binary_mask(smp), size, mode="any").astype(np.uint8))
    return EvalSet(name, list(samples), grids, sizes, gts, [s.label for s in samples])


def _grid32(g):
    from src.feature_extraction.dinov2_extractor import PatchGrid

    return PatchGrid(g.features.astype(np.float32), g.input_size, g.source_size)


def heatmaps_for(es: EvalSet, bank: MemoryBank, cfg: PipelineConfig, exclude: list | None = None) -> list:
    exclude = exclude or [None] * len(es)
    return [upsample_heatmap(bank.score_grid(_grid32(g), ex), size, cfg.smooth_patches)
            for g, size, ex in zip(es.grids, es.sizes, exclude)]


# ------------------------------------------------------------------ heatmap-level
def run_heatmap_eval(es: EvalSet, heatmaps: list, include_normalized: bool = True) -> dict:
    labels = np.array(es.labels)
    out = {"n_images": len(es)}
    groups = {"all": np.arange(len(es))}
    for cls in sorted(set(es.labels)):
        groups[cls] = np.nonzero(labels == cls)[0]
    for name, idx in groups.items():
        hm = [heatmaps[i] for i in idx]
        gt = [es.gts[i] for i in idx]
        row = {f"raw_{k}": v for k, v in evaluate(hm, gt).items()}
        if include_normalized:  # the ORIGINAL (per-image min-max) protocol, for comparison only
            row.update({f"per_image_normalized_{k}": v
                        for k, v in evaluate([normalize_for_display(h) for h in hm], gt).items()})
        row["n"] = int(len(idx))
        out[name] = row
    return out


def run_image_level_eval(bank: MemoryBank, normal_loo_heatmaps: list, es: EvalSet, cfg: PipelineConfig,
                         calibration: Calibration, seed: int = 0) -> dict:
    """Negatives: the reference images scored leave-one-out (bank of n-1).
    Positives: defect images scored against a bank with one random reference image removed,
    so both sides see an (n-1)-image bank. Reported as a caveat: only n reference normals exist."""
    rng = np.random.default_rng(seed)
    ids = np.unique(bank.image_ids)
    excl = [int(rng.choice(ids)) for _ in range(len(es))]
    pos = [float(h.max()) for h in heatmaps_for(es, bank, cfg, excl)]
    neg = [float(h.max()) for h in normal_loo_heatmaps]
    scores, labels = neg + pos, [0] * len(neg) + [1] * len(pos)
    return {
        "image_auroc": image_auroc(scores, labels),
        "at_calibrated_threshold": image_decision_metrics(scores, labels, calibration.image_threshold),
        "n_normal": len(neg), "n_defect": len(pos),
        "note": "Normals are the reference images scored leave-one-out; only 20 exist in MSD.",
    }


# ------------------------------------------------------------------ B: thresholds
def exp_B_thresholds(es_val: EvalSet, val_heatmaps: list, normal_loo: list, cfg: PipelineConfig,
                     percentiles=(95, 98, 99, 99.5, 99.8, 99.9)) -> dict:
    rows = []
    base = calibrate_from_normals(normal_loo, None, cfg.pixel_percentile, cfg.image_rule, cfg.image_margin,
                                  cfg.image_k_std)

    def eval_low(low, tag):
        masks, n_regions = [], []
        for h, size, g in zip(val_heatmaps, es_val.sizes, es_val.grids):
            ppp = px_per_patch(g.grid_shape, size)
            r, m = extract_regions(h, low, base.image_threshold, None,
                                   int(round(cfg.min_area_patches * ppp * ppp)), cfg.max_regions)
            masks.append(m)
            n_regions.append(len(r))
        met = binary_segmentation_metrics(masks, es_val.gts)
        rows.append({"method": tag, "pixel_threshold": float(low),
                     "mean_regions_per_image": float(np.mean(n_regions)), **met})

    for p in percentiles:
        eval_low(float(np.percentile(np.concatenate([h.ravel() for h in normal_loo]), p)), f"loo_normals_p{p}")
    eval_low(calibrate_pixel_from_background(val_heatmaps, es_val.gts, cfg.pixel_percentile),
             f"val_background_p{cfg.pixel_percentile} (uses labels)")
    t, d = tune_pixel_threshold_on_val(val_heatmaps, es_val.gts)
    eval_low(t, "val_tuned_max_dice (uses labels, upper bound)")
    return {"image_threshold": base.image_threshold, "rows": rows}


# ------------------------------------------------------------------ C: peaks vs regions
def _prompt_hits(points_xy, gt, tol: int = 5) -> float:
    if not len(points_xy):
        return float("nan")
    from scipy.ndimage import binary_dilation

    g = binary_dilation(gt.astype(bool), iterations=tol)
    return float(np.mean([g[int(y), int(x)] for x, y in points_xy]))


def exp_C_peaks_vs_regions(es: EvalSet, heatmaps: list, normal_loo: list, cfg: PipelineConfig,
                           calibration: Calibration, legacy_thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> dict:
    """Legacy: peaks on per-image-normalized heatmaps. New: calibrated regions on raw heatmaps.
    Reports prompts per image, fraction of prompts on a real defect, GT-region recall, and
    how often a CLEAN reference image (leave-one-out) would get prompted/flagged."""
    rows = []
    for t in legacy_thresholds:
        n, hit, rec = [], [], []
        for h, gt in zip(heatmaps, es.gts):
            pk = find_peaks(normalize_for_display(h), threshold=t)
            n.append(len(pk))
            hit.append(_prompt_hits([(p.x, p.y) for p in pk], gt))
            pm = np.zeros_like(gt)
            for p in pk:
                pm[p.y, p.x] = 1
            rec.append(pm)
        clean_flag = np.mean([len(find_peaks(normalize_for_display(h), threshold=t)) > 0 for h in normal_loo])
        from scipy.ndimage import binary_dilation

        met = binary_segmentation_metrics([binary_dilation(m, iterations=3) for m in rec], es.gts,
                                          region_hit_frac=1e-9)
        rows.append({"method": f"legacy_peaks_norm@{t}", "prompts_per_image": float(np.mean(n)),
                     "prompt_precision": float(np.nanmean(hit)), "gt_region_recall": met["gt_region_recall"],
                     "clean_images_flagged": float(clean_flag)})

    n, hit, rec_masks = [], [], []
    for h, size, g, gt in zip(heatmaps, es.sizes, es.grids, es.gts):
        ppp = px_per_patch(g.grid_shape, size)
        r, m = extract_regions(h, calibration.pixel_threshold, calibration.image_threshold, None,
                               int(round(cfg.min_area_patches * ppp * ppp)), cfg.max_regions)
        n.append(len(r))
        hit.append(_prompt_hits([rg.peak_xy for rg in r], gt))
        rec_masks.append(m)
    clean_flag = np.mean([h.max() > calibration.image_threshold for h in normal_loo])
    met = binary_segmentation_metrics(rec_masks, es.gts)
    rows.append({"method": "calibrated_regions(raw)", "prompts_per_image": float(np.mean(n)),
                 "prompt_precision": float(np.nanmean(hit)), "gt_region_recall": met["gt_region_recall"],
                 "clean_images_flagged": float(clean_flag),
                 "note": "clean-image rate on LOO reference images is ~0 by construction when image_rule='max'"})
    return {"rows": rows}


# ------------------------------------------------------------------ D + F: masks
def _load_work_image(dataset, sample, size) -> Image.Image:
    return dataset.load_image(sample).resize(size, Image.BILINEAR)


def exp_D_F_masks(es: EvalSet, heatmaps: list, dataset, cfg: PipelineConfig, calibration: Calibration,
                  segmenter=None, strategies=("point", "multi_point", "box_point", "box_pos_neg"),
                  include_legacy: bool = True, limit: int | None = None, progress: bool = True) -> dict:
    """Binary-mask quality of: thresholded heatmap, legacy SAM2 (peaks@0.6 on normalized
    heatmap, unfiltered union) and each SAM 2 prompt strategy with calibrated regions."""
    idx = list(range(len(es)))[:limit] if limit else list(range(len(es)))
    methods = {"heatmap_regions": []}
    if segmenter is not None:
        if include_legacy:
            methods["legacy_sam2_peaks@0.6"] = []
        for s in strategies:
            methods[f"sam2_{s}"] = []
    fallbacks = {k: [] for k in methods}
    labels = []
    t0 = time.time()
    for i in _progress(idx, "masks", progress):
        h, size, g, gt = heatmaps[i], es.sizes[i], es.grids[i], es.gts[i]
        labels.append(es.labels[i])
        ppp = px_per_patch(g.grid_shape, size)
        regions, hmask = extract_regions(h, calibration.pixel_threshold, calibration.image_threshold, None,
                                         int(round(cfg.min_area_patches * ppp * ppp)), cfg.max_regions)
        methods["heatmap_regions"].append(hmask)
        if segmenter is None:
            continue
        img = _load_work_image(dataset, es.samples[i], size)
        if include_legacy:
            pk = find_peaks(normalize_for_display(h), threshold=0.6)
            methods["legacy_sam2_peaks@0.6"].append(segmenter.segment(img, pk).mask)
        for s in strategies:
            res = segmenter.segment_regions(img, regions, h, calibration.pixel_threshold, s, cfg.n_pos, cfg.n_neg,
                                            cfg.box_pad_frac, cfg.min_containment,
                                            int(round(cfg.containment_dilate_patches * ppp)))
            methods[f"sam2_{s}"].append(res.mask)
            fallbacks[f"sam2_{s}"].append((res.n_fallbacks, len(regions)))

    gts = [es.gts[i] for i in idx]
    out = {"n_images": len(idx), "elapsed_s": round(time.time() - t0, 1), "calibration": calibration.method,
           "methods": {}}
    for name, masks in methods.items():
        if not masks:
            continue
        row = {"all": binary_segmentation_metrics(masks, gts)}
        for cls in sorted(set(labels)):
            sel = [j for j, l in enumerate(labels) if l == cls]
            row[cls] = binary_segmentation_metrics([masks[j] for j in sel], [gts[j] for j in sel])
        if fallbacks.get(name):
            fb, tot = np.array(fallbacks[name]).sum(0)
            row["sam_fallback_rate"] = float(fb / max(tot, 1))
        out["methods"][name] = row
    return out


# ------------------------------------------------------------------ E: bank size
def exp_E_bank_size(es: EvalSet, full_bank: MemoryBank, normal_grids: list, normal_sizes: list,
                    cfg: PipelineConfig, sizes=(3, 5, 10, 15, 20), repeats: int = 3, seed: int = 0) -> dict:
    """For n < 20 the unused reference images are genuinely held-out normals, giving an
    honest false-alarm rate on clean screens that the 20-image setting cannot provide."""
    rng = np.random.default_rng(seed)
    all_ids = np.unique(full_bank.image_ids)
    rows = []
    for n in sizes:
        reps = 1 if n >= len(all_ids) else repeats
        for r in range(reps):
            chosen = sorted(rng.choice(all_ids, n, replace=False).tolist()) if n < len(all_ids) else all_ids.tolist()
            held = [i for i in all_ids if i not in chosen]
            bank = full_bank.subset(chosen)
            loo = [upsample_heatmap(bank.score_grid(normal_grids[i], exclude_image=i), normal_sizes[i],
                                    cfg.smooth_patches) for i in chosen]
            cal = calibrate_from_normals(loo, None, cfg.pixel_percentile, cfg.image_rule, cfg.image_margin,
                                         cfg.image_k_std)
            hm = heatmaps_for(es, bank, cfg)
            row = {"n_reference": n, "repeat": r, **{f"raw_{k}": v for k, v in evaluate(hm, es.gts).items()}}
            dscores = [float(h.max()) for h in hm]
            row["defect_images_detected"] = float(np.mean(np.array(dscores) > cal.image_threshold))
            if held:
                hs = [float(upsample_heatmap(bank.score_grid(normal_grids[i]), normal_sizes[i],
                                             cfg.smooth_patches).max()) for i in held]
                row["heldout_normals_flagged"] = float(np.mean(np.array(hs) > cal.image_threshold))
                row["image_auroc_heldout"] = image_auroc(hs + dscores, [0] * len(hs) + [1] * len(dscores))
                row["n_heldout_normals"] = len(held)
            rows.append(row)
    return {"rows": rows}


# ------------------------------------------------------------------ A: live ROI
def exp_A_roi(inspector, photos: list, labels: list | None = None, manual_boxes: list | None = None,
              methods=("full", "manual", "auto")) -> dict:
    """labels: optional 'clean'/'defect' per photo. Without GT masks, this measures what
    matters for the finger problem: how often clean photos are flagged and how much area
    is flagged. Returns records plus the InspectionResults for plotting."""
    rows, results = [], []
    saved = inspector.roi_method
    try:
        for i, photo in enumerate(photos):
            for m in methods:
                if m == "manual" and (not manual_boxes or manual_boxes[i] is None):
                    continue
                inspector.roi_method = None if m == "full" else ("manual" if m == "manual" else "auto")
                kw = {"manual_box": manual_boxes[i]} if m == "manual" else None
                res = inspector.inspect(photo, roi_kwargs=kw)
                results.append((i, m, res))
                rows.append({"photo": i, "label": labels[i] if labels else None, "method": m,
                             "roi_used": getattr(res.roi, "method", "none"),
                             "is_defective": bool(res.is_defective), "image_score": res.image_score,
                             "n_regions": len(res.regions), "flagged_area_pct": 100.0 * float(res.final_mask.mean())})
    finally:
        inspector.roi_method = saved
    summary = {}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            continue
        s = {"n": len(rs), "flagged_rate": float(np.mean([r["is_defective"] for r in rs])),
             "mean_flagged_area_pct": float(np.mean([r["flagged_area_pct"] for r in rs]))}
        if labels:
            clean = [r for r in rs if r["label"] == "clean"]
            dfct = [r for r in rs if r["label"] == "defect"]
            if clean:
                s["clean_false_alarm_rate"] = float(np.mean([r["is_defective"] for r in clean]))
            if dfct:
                s["defect_detection_rate"] = float(np.mean([r["is_defective"] for r in dfct]))
        summary[m] = s
    return {"rows": rows, "summary": summary, "_results": results}
