# Zero-Shot & Few-Shot Defect Segmentation for Smartphone Screens
DINOv2 memory bank + calibrated decisions + SAM 2 — Industrial Optical Inspection Capstone
(BTech AI, Sem VII — Group 9: Prit Desai I017, Yesha Doshi I021)

**v2** fixes every issue from the code review; see [`docs/CHANGES.md`](docs/CHANGES.md) for the
issue → fix → file map.

## Pipeline

```
[live only] screen ROI (SAM 2 centre prompt → rectangle → perspective warp, fingers excluded)
  → DINOv2 patch features (aspect-preserving, 74×42 grid for 16:9)
  → k-NN cosine distance to a memory bank of normal patches (k = 1, raw scores)
  → heatmap at 960 px (invalid patches neutralized before smoothing)
  → image score = max → DEFECTIVE iff > image threshold      ← calibrated, label-free
  → hysteresis regions (pixel threshold … image threshold)
  → SAM 2 per region: box + positive + negative points, multimask,
    mask accepted only if it lies inside the heatmap region, else fallback
  → final mask
```

Thresholds come from **leave-one-out** scores of the clean reference images (each scored against the
others), so the pipeline can say "this screen is normal". The reference set is either MSD's 20 `good/`
images (JPEG-matched to the defect images automatically) or 5–10 photos of **your** demo phone
(few-shot mode).

## Project structure

```
src/
  config.py               PipelineConfig: every setting, saved into every results file
  data/                   MSD loader: stratified split, JPEG-quality estimation, JPEG round-trip
  feature_extraction/     DINOv2 (aspect/square resize, _reg variants, intermediate layers, batching)
  memory_bank/            k-NN bank with image ids (leave-one-out), patch exclusion, save/load + config check
  heatmap/                raw-score upsampling, patch-unit smoothing, display-only normalization
  calibration/            label-free LOO thresholds, label-using alternatives, domain-shift diagnostic
  postprocess/            hysteresis regions, validity masks
  sam2_prompt/            SAM 2 wrapper, 4 prompt strategies, containment-based mask selection
  roi/                    screen detection (SAM 2 / OpenCV / manual) + warp + validity mask
  evaluation/             pixel AUROC, PRO, image AUROC, IoU/Dice/precision/recall/region recall
  inference.py            DefectInspector + builders (MSD reference / few-shot)
  experiments.py          evaluation runner + experiments A–G
  visualize.py            figures
  run_pipeline.py         CLI
tests/                    mocks + 19 tests (no GPU needed): python -m tests.test_pipeline
notebooks/capstone_pipeline.ipynb   phase-wise Colab notebook (Phases 0–6)
docs/CHANGES.md           issue → fix → file
```

## Dataset

**MSD** (Zhang & Ban, ICASSP 2022): 1,200 defect images (400 each of oil / scratch / stain) at
1920×1080 with pixel masks, plus 20 defect-free images. The split is a **seeded random, class-stratified
60/20/20 split made for this project** (not the official FDSNet split). The val split is used for the
diagnostic, threshold study and feature ablation; the test split only for reported numbers.

## Running it

**Colab (recommended):** zip the project folder with `src/` and `data/raw/MSD/`, open
`notebooks/capstone_pipeline.ipynb` on a T4 GPU, and run Phases 0–6 in order. Each phase writes
`results/phaseN_*.json`; Phase 6 packs everything into `results_bundle.zip`.

**CLI:**
```
pip install -r requirements.txt && pip install "git+https://github.com/facebookresearch/sam2.git"
python -m src.run_pipeline --msd-root data/raw/MSD --split test          # improved pipeline
python -m src.run_pipeline --msd-root data/raw/MSD --skip-sam2 --baseline # original settings
```

**Tests (no GPU):** `python -m tests.test_pipeline`

## Experiments

| | Question | Split |
|---|---|---|
| A | Full image vs manual crop vs automatic ROI on live photos | live |
| B | Which pixel threshold (label-free vs val-tuned upper bound)? | val |
| C | Legacy peaks@0.6 vs calibrated regions (incl. clean-screen false alarms) | test |
| D | SAM 2 prompt strategy | test |
| E | Number of reference images (with truly held-out clean screens) | test |
| F | Heatmap vs thresholded heatmap vs SAM 2 masks (+ legacy SAM 2) | test |
| G | Feature settings: square vs aspect, k, registers, layers, JPEG matching | val |

## Notes for the viva

* The old numbers (AUROC ≈ 0.99) were computed on per-image-normalized maps. v2 reports raw-score
  numbers next to that protocol so the effect is visible.
* Thresholds are in **raw cosine-distance units** from leave-one-out calibration, not "0.6".
* k = 1 is the PatchCore/SPADE nearest-neighbour score. PatchCore also uses coreset subsampling, which
  is unnecessary at this bank size.
* Only 20 clean MSD images exist, so image-level results on MSD rest on 20 negatives (leave-one-out).
  State this limitation; Experiment E adds truly held-out clean screens.
* "Zero-shot" = no defect labels for training or thresholds. The val-tuned threshold in Experiment B
  uses labels and is reported only as an upper bound.

## Status

- [x] Dataset, v1 pipeline, v1 results
- [x] Code review: 6 critical + 12 minor issues
- [x] v2 rewrite, all issues fixed, 19/19 tests passing on mocks
- [x] Phase-wise Colab notebook
- [ ] Run Phases 0–6 on Colab → `results_bundle.zip`
- [ ] Report + viva slides from the real v2 results
