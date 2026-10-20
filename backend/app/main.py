
"""Phase 3: Mobile Defect Detection API — image upload and validation."""

from __future__ import annotations

import io
import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 30_000_000

FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}

SUFFIX = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}

STORAGE = Path(
    os.environ.get(
        "UPLOAD_DIR",
        str(Path(__file__).resolve().parents[2] / "data" / "uploads"),
    )
)

app = FastAPI(
    title="Mobile Defect Detection API",
    version="0.3.1",
    description="Image upload API. Model inference will be integrated later.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class UploadReceipt(BaseModel):
    upload_id: str
    status: str
    original_filename: str
    media_type: str
    size_bytes: int
    width: int
    height: int
    inference_performed: bool


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "service": "mobile-defect-api",
        "model_loaded": False,
        "inference_available": False,
    }


@app.post(
    "/api/v1/uploads",
    response_model=UploadReceipt,
    status_code=201,
)
async def upload_image(file: UploadFile = File(...)):

    # Step 1: Read uploaded file
    contents = await file.read(MAX_BYTES + 1)
    await file.close()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit.",
        )

    # Step 2: Detect actual image format using Pillow
    try:
        with Image.open(io.BytesIO(contents)) as im:

            actual_format = im.format
            width, height = im.size

            if actual_format not in FORMAT_TO_MIME:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported image format: {actual_format}",
                )

            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Image has invalid dimensions.",
                )

            if width * height > MAX_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions exceed the 30-megapixel limit.",
                )

            # Verify image file integrity
            im.verify()

    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupt image file.",
        )

    # Step 3: Use actual format, rather than trusting MIME metadata
    detected_mime = FORMAT_TO_MIME[actual_format]

    # Step 4: Generate unique filename
    upload_id = uuid4().hex

    STORAGE.mkdir(parents=True, exist_ok=True)

    path = STORAGE / f"{upload_id}{SUFFIX[actual_format]}"

    # Step 5: Save image
    try:
        with path.open("xb") as output:
            output.write(contents)

    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to save uploaded image.",
        ) from exc

    # Step 6: Return metadata
    return UploadReceipt(
        upload_id=upload_id,
        status="uploaded",
        original_filename=Path(
            (file.filename or "image").replace("\\", "/")
        ).name,
        media_type=detected_mime,
        size_bytes=len(contents),
        width=width,
        height=height,
        inference_performed=False,
    )
