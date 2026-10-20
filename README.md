# Mobile Defect Detection Studio

Full-stack smartphone screen defect detection project, built incrementally.

## Current status

Phase 1: repository scaffold. No web UI or API inference is implemented yet.

## Existing research pipeline

The original Python pipeline is in `src/`, with the original Colab notebook in `notebooks/capstone_pipeline.ipynb`. It uses DINOv2 features, a clean-reference memory bank, calibrated anomaly scoring, and optional SAM 2 segmentation.

Read `docs/ORIGINAL_PROJECT_README.md` for the original pipeline's requirements and reported project scope; see `docs/CHANGES.md` for historical changes. The original dependencies are in `requirements-model.txt`.

## Planned application

- `frontend/`: HTML, Tailwind CSS, JavaScript, Vite (Phase 2).
- `backend/`: FastAPI inference services (Phases 3–4).
- `results/`: local experimental evaluation outputs (Phase 5 onward).
- `artifacts/`: local model and memory-bank artifacts.
- `data/`: local datasets; never committed to Git.

## Reproducibility

Model weights and large datasets are not included. Full model execution depends on installing Python model dependencies, downloading weights, building calibrated reference-bank artifacts and preparing the image datasets. Google Colab T4 is used for GPU-heavy experiments.

## Phase 1 verification

From project root, run:

```powershell
python -m compileall -q src
git status
```
