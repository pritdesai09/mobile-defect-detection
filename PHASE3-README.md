# Phase 3 update instructions

Extract this ZIP to `C:\Users\Prit\mobile-defect-detection` using Expand-Archive -Force. It contains only `frontend/` and `backend/`, so the existing `src/`, notebooks, and results are untouched.

Add these lines to your root `.gitignore` if absent:

```gitignore
/data/uploads/
```

Then install backend dependencies, run uvicorn on port 8000, and run the Vite frontend on port 5173 in a second terminal. Test `GET /api/v1/health` and `POST /api/v1/uploads` from the browser UI. No AI inference is performed in this phase.
