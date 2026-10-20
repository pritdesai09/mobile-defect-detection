# Phase 3 FastAPI

Run from repository root (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Visit http://127.0.0.1:8000/docs . Uploaded images are saved locally to `data/uploads/` and are excluded from Git. This endpoint **does not perform inference**.
