# Phase 4: DINOv2 model integration

1. **Local**: merge ZIP into existing repository root. Preserve your own backend fixes if you made further changes; archive includes the corrected MIME behavior.
2. Install local API: `.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt`.
3. Start API from repository root: `.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000`.
4. Before artifacts are exported, the inference endpoint should give HTTP 503, NOT a dummy prediction.
5. **Colab**: Run the steps in the chat. A GPU is needed for the reference bank export. Transfer the *same* `src/` package to Colab and ensure the MSD folder path matches the loader.
6. Copy `memory_bank.npz`, `calibration.json`, `config.json` to local `artifacts/msd/`.
7. Local inference needs model libraries from `backend/requirements-model-local.txt` and may download DINOv2 weights on the first request. CPU will be much slower than T4.
8. Optional SAM 2: must install SAM 2 and provide matching SAM2_CHECKPOINT and SAM2_CONFIG. Set ENABLE_SAM2=1 before launching API. No checkpoint is included.
9. Use `npm run build` under frontend and test GET health, POST uploads, POST inspections/{id}, GET images.
10. No dataset, downloaded checkpoint or derived results should be committed.
