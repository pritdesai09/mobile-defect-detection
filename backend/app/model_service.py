"""Lazy, artifact-backed DINOv2 defect inspector. No synthetic predictions."""
from __future__ import annotations
import io, json, os, threading
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path(os.environ.get('MODEL_ARTIFACT_DIR', str(ROOT / 'artifacts' / 'msd')))
_lock = threading.Lock()
_inspector = None
_error = None

def configured():
    return all((ARTIFACT_DIR / name).is_file() for name in ('memory_bank.npz', 'calibration.json', 'config.json'))

def status():
    return {'artifacts_present': configured(), 'model_loaded': _inspector is not None,
            'inference_available': configured(), 'model_error': _error,
            'sam_enabled': bool(_inspector is not None and _inspector.segmenter is not None)}

def get_inspector():
    global _inspector, _error
    if _inspector is not None: return _inspector
    if not configured(): raise RuntimeError('Model artifacts missing. Export them from Colab first.')
    with _lock:
        if _inspector is not None: return _inspector
        try:
            from src.config import PipelineConfig
            from src.calibration.calibration import Calibration
            from src.memory_bank.memory_bank import MemoryBank
            from src.feature_extraction.dinov2_extractor import DinoV2Extractor
            from src.inference import DefectInspector
            cfg = PipelineConfig.from_dict(json.loads((ARTIFACT_DIR/'config.json').read_text()))
            cal = Calibration.from_dict(json.loads((ARTIFACT_DIR/'calibration.json').read_text()))
            extractor = DinoV2Extractor.from_config(cfg)
            bank = MemoryBank.load(str(ARTIFACT_DIR/'memory_bank.npz'))
            segmenter = None
            # SAM 2 requires installation and a matching checkpoint/config.
            if os.environ.get('ENABLE_SAM2', '0') == '1':
                from src.sam2_prompt.sam2_predictor import Sam2DefectSegmenter
                ckpt = os.environ.get('SAM2_CHECKPOINT')
                config = os.environ.get('SAM2_CONFIG', 'configs/sam2.1/sam2.1_hiera_s.yaml')
                if not ckpt or not Path(ckpt).is_file():
                    raise RuntimeError('ENABLE_SAM2=1 requires a valid SAM2_CHECKPOINT path')
                segmenter = Sam2DefectSegmenter(ckpt, config)
            # MSD images already contain screen imagery; use full-image mode by default.
            _inspector = DefectInspector(extractor, bank, cal, cfg, segmenter=segmenter, roi_method=None)
            _error = None
            return _inspector
        except Exception as exc:
            _error = str(exc)
            raise

def inspect_path(path: Path, output_dir: Path, upload_id: str):
    import cv2
    inspector = get_inspector()
    with Image.open(path) as opened:
        image = opened.convert('RGB')
    # Model objects can be unsafe under concurrent requests; serialize inference.
    with _lock:
        result = inspector.inspect(image, run_sam=(inspector.segmenter is not None))
    output_dir.mkdir(parents=True, exist_ok=True)
    w,h = result.image.size
    heat = np.asarray(result.heatmap, dtype=np.float32)
    valid = np.asarray(result.valid_mask, dtype=bool)
    vals=heat[valid & np.isfinite(heat)]
    if vals.size:
        low=float(np.quantile(vals,0.05));high=float(np.quantile(vals,0.99))
        scale=max(high-low,1e-8)
        display=np.clip((heat-low)/scale,0,1)
    else: display=np.zeros_like(heat)
    # Normalization here is ONLY for visualization; never for score/threshold.
    heat_rgb=cv2.cvtColor(cv2.applyColorMap((display*255).astype('uint8'),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    mask=(np.asarray(result.final_mask)>0).astype('uint8')*255
    mask_img=Image.fromarray(mask).convert('RGB')
    overlay=np.asarray(result.image.convert('RGB')).copy()
    overlay[mask>0]=(0.55*overlay[mask>0]+0.45*np.array([250,55,65])).astype('uint8')
    names={'heatmap': f'{upload_id}_heatmap.png', 'mask':f'{upload_id}_mask.png', 'overlay':f'{upload_id}_overlay.png'}
    Image.fromarray(heat_rgb).save(output_dir/names['heatmap'])
    mask_img.save(output_dir/names['mask'])
    Image.fromarray(overlay).save(output_dir/names['overlay'])
    finite_score = float(result.image_score) if np.isfinite(result.image_score) else None
    return {'upload_id':upload_id, 'status':'completed', 'prediction':'defective' if result.is_defective else 'normal',
      'image_score':finite_score, 'image_threshold':float(result.calibration.image_threshold),
      'pixel_threshold':float(result.calibration.pixel_threshold), 'n_regions':len(result.regions),
      'mask_area_percent':round(float((mask>0).mean()*100),3),
      'sam_used':result.sam_mask is not None, 'roi_used':result.roi is not None,
      'warning':result.info.get('warning'), 'work_width':w, 'work_height':h,
      'artifacts':{k:f'/api/v1/inspections/{upload_id}/{k}' for k in names}}
