"""FastAPI upload receipt + genuine, artifact-gated inference."""
from __future__ import annotations
import io, os
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from backend.app import model_service
MAX_BYTES=10*1024*1024
MAX_PIXELS=30_000_000
MIMES={'JPEG':'image/jpeg','PNG':'image/png','WEBP':'image/webp'}
EXT={'JPEG':'.jpg','PNG':'.png','WEBP':'.webp'}
ROOT=Path(__file__).resolve().parents[2]
STORAGE=Path(os.environ.get('UPLOAD_DIR',str(ROOT/'data'/'uploads')))
OUTPUTS=Path(os.environ.get('INSPECTION_OUTPUT_DIR',str(ROOT/'data'/'inspections')))
app=FastAPI(title='Mobile Defect Detection API',version='0.4.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_credentials=False,allow_methods=['GET','POST'],allow_headers=['Content-Type'])
class UploadReceipt(BaseModel):
    upload_id:str
    status:str
    original_filename:str
    media_type:str
    size_bytes:int
    width:int
    height:int
    inference_performed:bool
@app.get('/api/v1/health')
def health():
    return {'status':'ok','service':'mobile-defect-api',**model_service.status()}
async def read_image(file:UploadFile):
    data=await file.read(MAX_BYTES+1)
    await file.close()
    if not data: raise HTTPException(400,'The uploaded file is empty.')
    if len(data)>MAX_BYTES: raise HTTPException(413,'Image exceeds the 10 MB limit.')
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt=im.format
            w,h=im.size
            if fmt not in MIMES: raise HTTPException(415,f'Unsupported image format: {fmt}')
            if w<=0 or h<=0: raise HTTPException(400,'Invalid image dimensions.')
            if w*h>MAX_PIXELS: raise HTTPException(413,'Image exceeds the 30 MP limit.')
            im.verify()
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError) as exc:
        raise HTTPException(400,'Invalid or corrupt image file.') from exc
    return data,fmt,w,h
@app.post('/api/v1/uploads',response_model=UploadReceipt,status_code=201)
async def upload_image(file:UploadFile=File(...)):
    data,fmt,w,h=await read_image(file)
    uid=uuid4().hex
    STORAGE.mkdir(parents=True,exist_ok=True)
    try:
        with (STORAGE/f'{uid}{EXT[fmt]}').open('xb') as f:f.write(data)
    except OSError as exc:raise HTTPException(500,'Unable to save image.') from exc
    return UploadReceipt(upload_id=uid,status='uploaded',original_filename=Path((file.filename or 'image').replace(chr(92),'/')).name,
       media_type=MIMES[fmt],size_bytes=len(data),width=w,height=h,inference_performed=False)
@app.post('/api/v1/inspections/{upload_id}')
def inspect_uploaded(upload_id:str):
    if len(upload_id)!=32 or not all(c in '0123456789abcdef' for c in upload_id):
        raise HTTPException(400,'Invalid upload ID.')
    match=next((STORAGE/f'{upload_id}{suffix}' for suffix in EXT.values() if (STORAGE/f'{upload_id}{suffix}').is_file()),None)
    if match is None:raise HTTPException(404,'Upload not found.')
    if not model_service.configured():raise HTTPException(503,'Model artifacts are not installed. Run the Colab export step.')
    try:return model_service.inspect_path(match,OUTPUTS,upload_id)
    except Exception as exc:
        # Internal details are logged server-side; avoid leaking paths to the client.
        import logging
        logging.exception('Inference failed for %s',upload_id)
        raise HTTPException(503,'Model inference is unavailable; check server logs and dependencies.') from exc
@app.get('/api/v1/inspections/{upload_id}/{kind}')
def inspection_image(upload_id:str,kind:str):
    if len(upload_id)!=32 or not all(c in '0123456789abcdef' for c in upload_id):raise HTTPException(400,'Invalid upload ID.')
    if kind not in {'heatmap','mask','overlay'}:raise HTTPException(404,'Unknown artifact type.')
    path=OUTPUTS/f'{upload_id}_{kind}.png'
    if not path.is_file():raise HTTPException(404,'Artifact not found.')
    return FileResponse(path,media_type='image/png',headers={'Cache-Control':'no-store'})
