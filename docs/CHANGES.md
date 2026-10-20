# What changed in v2, and why

Every issue found in the code review, what was done about it, and where the fix lives.
"Verified" = covered by `tests/test_pipeline.py` (mock models, synthetic data). Real-data numbers come
only from running the notebook on Colab.

## Critical issues

| # | Issue | Fix | Where | Verified |
|---|---|---|---|---|
| 1 | Per-image min-max normalization + threshold 0.6 ⇒ the maximum is always 1.0 ⇒ **every image, even a clean one, was "defective"** and SAM 2 was always prompted. | Heatmaps stay in **raw cosine-distance units**. Two thresholds are calibrated **label-free** from leave-one-out scores of the normal reference images: image threshold = highest score on any clean screen; pixel threshold = 99.5th pct of clean pixels. Regions use hysteresis (seed > image thr, grow > pixel thr), so no detection ⇒ no regions ⇒ no SAM prompts. | `calibration/calibration.py`, `postprocess/regions.py`, `inference.py` | ✅ clean screens pass, defects flagged; legacy behaviour reproduced as a test |
| 2 | Reported metrics were computed on per-image-normalized maps; test set had no normal images, so no image-level metric. | Metrics on raw scores, reported **next to** the old protocol. Image-level AUROC with leave-one-out normals; decision rates at the calibrated threshold. Experiment E adds genuinely held-out clean screens. | `experiments.py` (`run_heatmap_eval`, `run_image_level_eval`, `exp_E_bank_size`) | ✅ |
| 3 | Live test is out of distribution (other phone/camera, fingers, background). | (a) **Screen ROI**: SAM 2 centre-point prompt → most rectangular mask → perspective warp to landscape; OpenCV/manual fallbacks. Fingers entering from the edge are excluded; internal holes (possible defects) are kept. (b) **Few-shot bank** from 5–10 clean photos of the demo phone, same ROI step, finger patches left out, LOO calibration. | `roi/screen_roi.py`, `inference.build_few_shot_inspector` | ✅ synthetic phone photos with fingers |
| 4 | Square 518×518 resize distorts 16:9 input (patch ≈ 52×29 px). | Aspect-preserving input (1920×1080 → 1036×588, 74×42 grid). Register models and intermediate layers selectable; ablation on **val** chooses. | `feature_extraction/dinov2_extractor.py`, notebook Phase 4 | ✅ size logic; features need Colab |
| 5 | SAM 2 masks never evaluated. | IoU / Dice / precision / recall (pooled + per image), GT-region recall, false-positive components, SAM fallback rate — heatmap vs legacy SAM 2 vs every prompt strategy. | `evaluation/metrics.py`, `experiments.exp_D_F_masks` | ✅ |
| 6 | Single positive point, `multimask_output=False`, no filtering. | Four strategies (point, multi-point via farthest-point sampling, box+point, **box + pos + neg**). Multimask on; a mask is accepted only if ≥ 60 % of it lies inside the dilated heatmap region (rejects whole-screen / finger masks), best SAM score wins; else fallback to the heatmap region (counted). | `sam2_prompt/sam2_predictor.py` | ✅ mock SAM whose top candidate is the whole image |

## Problems found while testing the rewrite

| Problem | Fix |
|---|---|
| **PNG normals vs JPEG defect images**: in testing, background of every defect image scored above anything seen on normals (median background at the 100th percentile of clean scores), so the entire screen became one "defect". MSD has exactly this structure. | Defect images' JPEG quality is estimated from their quantization tables (exact for q 60–98) and the normals are re-encoded to match (`normal_jpeg_quality="auto"`). A diagnostic reports where typical background falls in the clean-score distribution and warns above the 90th percentile. After the fix: 75.6th percentile in the synthetic test. |
| Finger scores (~25× threshold) bled through bilinear upsampling + Gaussian smoothing into valid pixels next to the finger. | Invalid patches are replaced by the median valid score **before** upsampling; patches straddling an occlusion edge are excluded (from the bank and from scoring). One shared `validity()` rule for reference and test images. |
| A finger in a reference photo put finger-edge patches into the bank and pushed the image threshold to ~25× normal. | Same strict patch validity when building the bank. |
| SAM screen masks with holes would have **hidden defects inside the screen** from scoring. | Internal holes are filled; only occluders touching the screen boundary stay invalid. |
| The diagnostic called pixels 5 px from a defect "background", but heatmap blobs legitimately extend 1–2 patches beyond a defect. | Uses a median-percentile test with a resolution-scaled exclusion band. |

## Minor issues

| Issue | Fix |
|---|---|
| Per-class and all-class splits were different random splits | All-class split = union of per-class splits (stratified); tested |
| Split described as "official" | Docstring and README say "seeded random 60/20/20 split made for this project" |
| Val split unused | Used for the diagnostic, Experiment B, and the feature ablation (Phase 4) |
| k = 9 described as PatchCore | Default k = 1 (PatchCore/SPADE nearest neighbour); k = 9 kept in the ablation |
| Gaussian sigma = 4 px negligible at 1920 px | Sigma defined in patch units (0.5 patch) at the working resolution |
| Heavy evaluation memory / slow PRO | Working resolution 960 px; GT downsampled with "any" pooling so thin scratches survive; histogram-based AUROC/PRO (checked against sklearn and a brute-force PRO) |
| `np.trapz` deprecated | `np.trapezoid` with fallback |
| PRO not interpolated at FPR = 0.3 | Curve starts at (0,0), interpolated at 0.3, trapezoid, normalized |
| Webcam cell text mismatch / no capture button | Capture-button webcam + gallery upload |
| Unused `data/memory_bank_normal/` | Banks saved with `MemoryBank.save()` together with the extractor config; loading into a mismatched extractor raises |
| SAM 2 config-name brittleness | Tries the given name, then `configs/sam2.1/<name>` |
