"""
CLI: calibrated evaluation on MSD.

    python -m src.run_pipeline --msd-root data/raw/MSD --split test \\
        --sam2-checkpoint checkpoints/sam2.1_hiera_small.pt --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml

    python -m src.run_pipeline --msd-root data/raw/MSD --skip-sam2 --baseline   # reproduce the original settings

Writes results/eval_summary.json containing: config, calibration, heatmap
metrics (raw AND the old per-image-normalized protocol), image-level metrics,
and binary-mask metrics for the thresholded heatmap and SAM 2.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.config import BASELINE_CONFIG, PipelineConfig
from src.data.msd_dataset import DEFECT_CLASSES, MSDDataset
from src.experiments import (
    build_eval_set,
    exp_D_F_masks,
    heatmaps_for,
    run_heatmap_eval,
    run_image_level_eval,
    save_json,
)
from src.inference import build_reference_inspector


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrated DINOv2 + memory bank + SAM 2 defect segmentation")
    ap.add_argument("--msd-root", default="data/raw/MSD")
    ap.add_argument("--defect-class", default="all", choices=["all", *DEFECT_CLASSES])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="cap images (quick runs)")
    ap.add_argument("--baseline", action="store_true", help="use the original square/k=9/point settings")
    ap.add_argument("--model", default=None, help="override model_name, e.g. dinov2_vits14_reg")
    ap.add_argument("--resize-mode", default=None, choices=["aspect", "square"])
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--prompt-strategy", default=None,
                    choices=["point", "multi_point", "box_point", "box_pos_neg"])
    ap.add_argument("--normal-jpeg-quality", default=None, help="'auto' (default), 'none', or an int")
    ap.add_argument("--skip-sam2", action="store_true")
    ap.add_argument("--sam2-checkpoint", default="checkpoints/sam2.1_hiera_small.pt")
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    ap.add_argument("--output-dir", default="results")
    a = ap.parse_args()

    cfg = BASELINE_CONFIG if a.baseline else PipelineConfig()
    overrides = {k: v for k, v in {"model_name": a.model, "resize_mode": a.resize_mode, "k": a.k,
                                   "prompt_strategy": a.prompt_strategy,
                                   "normal_jpeg_quality": a.normal_jpeg_quality}.items() if v is not None}
    q = overrides.get("normal_jpeg_quality")
    if q is not None:
        overrides["normal_jpeg_quality"] = None if q == "none" else (q if q == "auto" else int(q))
    cfg = cfg.with_(**overrides)

    from src.feature_extraction.dinov2_extractor import DinoV2Extractor

    ds = MSDDataset(a.msd_root)
    extractor = DinoV2Extractor.from_config(cfg)
    segmenter = None
    if not a.skip_sam2:
        from src.sam2_prompt.sam2_predictor import Sam2DefectSegmenter

        segmenter = Sam2DefectSegmenter(a.sam2_checkpoint, a.sam2_config)

    t0 = time.time()
    insp = build_reference_inspector(extractor, ds, cfg, segmenter)
    samples = ds.split(None if a.defect_class == "all" else a.defect_class, seed=cfg.seed)[a.split]
    samples = samples[: a.limit] if a.limit else samples
    es = build_eval_set(extractor, ds, samples, cfg, a.split)
    hm = heatmaps_for(es, insp.bank, cfg)

    summary = {
        "config": cfg.to_dict(),
        "split": a.split, "defect_class": a.defect_class, "n_images": len(es),
        "memory_bank": {"size": insp.bank.size, "embed_dim": insp.bank.embed_dim, "n_images": insp.bank.n_images},
        "calibration": insp.calibration.to_dict(),
        "heatmap_metrics": run_heatmap_eval(es, hm),
        "image_level": run_image_level_eval(insp.bank, insp.normal_loo_heatmaps, es, cfg, insp.calibration),
        "mask_metrics": exp_D_F_masks(es, hm, ds, cfg, insp.calibration, segmenter,
                                      strategies=(cfg.prompt_strategy,)),
    }
    summary["elapsed_s"] = round(time.time() - t0, 1)
    out = Path(a.output_dir) / "eval_summary.json"
    save_json(summary, out)

    hmm = summary["heatmap_metrics"]
    print("\n=== Heatmap (raw scores) ===")
    for k, v in hmm.items():
        if isinstance(v, dict):
            print(f"{k:>8s}  AUROC {v['raw_pixel_auroc']:.4f}  PRO {v['raw_pro_score']:.4f}   "
                  f"(old per-image-normalized: {v['per_image_normalized_pixel_auroc']:.4f} / "
                  f"{v['per_image_normalized_pro_score']:.4f})")
    il = summary["image_level"]
    print(f"Image AUROC {il['image_auroc']:.4f}; at threshold: {il['at_calibrated_threshold']}")
    for name, row in summary["mask_metrics"]["methods"].items():
        print(f"{name:>22s}  Dice {row['all']['pooled_dice']:.4f}  IoU {row['all']['pooled_iou']:.4f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
