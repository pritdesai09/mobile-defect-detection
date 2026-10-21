"""Run in Colab from repository root with MSD clean refs available."""
from pathlib import Path
import json, argparse
from src.config import PipelineConfig
from src.data.msd_dataset import MSDDataset
from src.feature_extraction.dinov2_extractor import DinoV2Extractor
from src.inference import build_reference_inspector

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--msd-root',required=True)
    p.add_argument('--out',default='artifacts/msd')
    a=p.parse_args()
    ds=MSDDataset(a.msd_root)
    cfg=PipelineConfig()
    extractor=DinoV2Extractor.from_config(cfg)
    inspector=build_reference_inspector(extractor,ds,cfg,segmenter=None)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    inspector.bank.save(str(out/'memory_bank.npz'))
    (out/'calibration.json').write_text(json.dumps(inspector.calibration.to_dict(),indent=2))
    (out/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2))
    print('Exported',out,'bank patches:',inspector.bank.size,'threshold:',inspector.calibration.image_threshold)
if __name__=='__main__':main()
