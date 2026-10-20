"""
End-to-end inspection: image -> normal/defective decision + location + mask.

    [optional] screen ROI  ->  DINOv2 patch grid  ->  raw k-NN scores
      -> heatmap at working resolution (raw units)  ->  validity mask
      -> image score = max valid heatmap value  ->  defective iff > image_threshold
      -> hysteresis regions (pixel_threshold .. image_threshold)
      -> [optional] SAM 2 per region with box/pos/neg prompts + containment check
      -> final mask

Two ways to build an inspector:
  build_reference_inspector(): memory bank + calibration from the MSD good/ images
  build_few_shot_inspector(): memory bank + calibration from a handful of clean
      photos of YOUR demo phone, taken the way you'll take the test photos. This
      is the "few-shot" mode, and it's what makes the live demo in-distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

from src.calibration.calibration import Calibration, calibrate_from_normals, loo_normal_heatmaps
from src.config import PipelineConfig
from src.heatmap.heatmap import neutralize_invalid, px_per_patch, upsample_heatmap, work_size
from src.memory_bank.memory_bank import MemoryBank
from src.postprocess.regions import (
    Region,
    border_valid_mask,
    extract_regions,
    patch_valid_strict,
    resize_mask,
)


@dataclass
class InspectionResult:
    image: Image.Image                 # image at working resolution (the ROI crop if ROI was used)
    heatmap: np.ndarray                # raw cosine-distance units, working resolution
    valid_mask: np.ndarray
    image_score: float
    is_defective: bool
    regions: list[Region]
    heatmap_mask: np.ndarray           # hysteresis-thresholded heatmap (uint8)
    sam_mask: np.ndarray | None
    final_mask: np.ndarray
    calibration: Calibration
    roi: object | None = None
    segmentation: object | None = None
    info: dict = field(default_factory=dict)

    def summary(self) -> str:
        if not self.is_defective:
            return (f"NORMAL: image score {self.image_score:.4f} <= threshold "
                    f"{self.calibration.image_threshold:.4f}")
        area = 100.0 * self.final_mask.mean()
        return (f"DEFECTIVE: image score {self.image_score:.4f} > threshold "
                f"{self.calibration.image_threshold:.4f}; {len(self.regions)} region(s), "
                f"{area:.2f}% of screen area")


class DefectInspector:
    def __init__(self, extractor, bank: MemoryBank, calibration: Calibration, cfg: PipelineConfig,
                 segmenter=None, roi_method: str | None = None):
        bank.check_compatible(getattr(extractor, "config", lambda: None)())
        self.extractor, self.bank, self.calibration, self.cfg = extractor, bank, calibration, cfg
        self.segmenter, self.roi_method = segmenter, roi_method

    def heatmap_for(self, image: Image.Image, exclude_image: int | None = None):
        """Returns (heatmap at working size, score grid, working size)."""
        grid = self.extractor.extract_patch_grid(image)
        score = self.bank.score_grid(grid, exclude_image)
        size = work_size(image.width, image.height, self.cfg.work_long_side)
        return upsample_heatmap(score, size, self.cfg.smooth_patches), score, size

    def inspect(self, image: Image.Image, valid_mask: np.ndarray | None = None, run_sam: bool = True,
                roi_kwargs: dict | None = None, calibration: Calibration | None = None) -> InspectionResult:
        cal = calibration or self.calibration
        cfg = self.cfg
        roi = None
        if self.roi_method:
            from src.roi.screen_roi import detect_screen

            kw = {"margin_frac": cfg.border_margin_frac or 0.03, "out_long_side": cfg.long_side}
            kw.update(roi_kwargs or {})
            roi = detect_screen(image, self.roi_method, self.segmenter, **kw)
            image, valid_mask = roi.image, roi.valid_mask

        grid = self.extractor.extract_patch_grid(image)
        score = self.bank.score_grid(grid)
        size = work_size(image.width, image.height, cfg.work_long_side)
        ppp = px_per_patch(score.shape, size)
        valid, patch_ok = validity(valid_mask, size, grid.grid_shape, cfg.border_margin_frac)
        heat = upsample_heatmap(neutralize_invalid(score, patch_ok), size, cfg.smooth_patches)
        img_w = image.convert("RGB").resize(size, Image.BILINEAR)

        image_score = float(heat[valid].max()) if valid.any() else float("-inf")
        is_def = image_score > cal.image_threshold
        min_area = int(round(cfg.min_area_patches * ppp * ppp))
        regions, hmask = extract_regions(heat, cal.pixel_threshold, cal.image_threshold, valid,
                                         min_area, cfg.max_regions)
        if not is_def:
            regions, hmask = [], np.zeros_like(hmask)

        sam_mask, seg = None, None
        if run_sam and self.segmenter is not None and regions:
            seg = self.segmenter.segment_regions(
                img_w, regions, heat, cal.pixel_threshold, cfg.prompt_strategy, cfg.n_pos, cfg.n_neg,
                cfg.box_pad_frac, cfg.min_containment,
                containment_dilate_px=int(round(cfg.containment_dilate_patches * ppp)), valid_mask=valid,
            )
            sam_mask = seg.mask
        final = sam_mask if sam_mask is not None else hmask
        info = {"px_per_patch": ppp, "min_area_px": min_area}
        if valid.any() and hmask[valid].mean() > 0.5:
            info["warning"] = ("Over half the screen exceeds the pixel threshold: the image probably differs "
                               "from the reference set (lighting, camera, JPEG). Recalibrate / use few-shot mode.")
        return InspectionResult(img_w, heat, valid, image_score, is_def, regions, hmask, sam_mask, final, cal,
                                roi, seg, info)


def validity(valid_mask: np.ndarray | None, size: tuple[int, int], grid_shape: tuple[int, int],
             border_margin_frac: float) -> tuple[np.ndarray, np.ndarray | None]:
    """Shared validity rule for reference images and test images.

    Returns (pixel mask at working size, patch mask or None). A patch is usable only if it is
    fully inside the valid area and not adjacent to an invalid patch (patches straddling a
    finger edge carry finger content). Pixel validity follows the patch mask.
    """
    pix = border_valid_mask((size[1], size[0]), border_margin_frac)
    if valid_mask is not None:
        pix &= resize_mask(valid_mask, size, mode="majority")
    if pix.all():
        return pix, None
    patch_ok = patch_valid_strict(pix, grid_shape)
    pix = pix & resize_mask(patch_ok, size, mode="majority")
    return pix, patch_ok


# ------------------------------------------------------------------ builders
def build_bank_and_calibration(extractor, images: list[Image.Image], cfg: PipelineConfig,
                               valid_masks: list[np.ndarray | None] | None = None,
                               group_ids: list[int] | None = None):
    """Memory bank + leave-one-out calibration from clean reference images.
    Returns (bank, calibration, grids, loo_heatmaps, valid_masks_at_work_size)."""
    grids = extractor.extract_batch(images)
    valid_masks = valid_masks or [None] * len(images)
    sizes = [work_size(im.width, im.height, cfg.work_long_side) for im in images]
    vp = [validity(v, s, g.grid_shape, cfg.border_margin_frac) for v, s, g in zip(valid_masks, sizes, grids)]
    vm_work, patch_ok = [x[0] for x in vp], [x[1] for x in vp]
    bank = MemoryBank(k=cfg.k).build(grids, patch_ok, group_ids,
                                     meta={"extractor": extractor.config(), "n_reference_images": len(images)})
    loo = loo_normal_heatmaps(bank, grids, sizes, cfg.smooth_patches, group_ids, patch_ok)
    cal = calibrate_from_normals(loo, vm_work, cfg.pixel_percentile, cfg.image_rule, cfg.image_margin,
                                 cfg.image_k_std)
    return bank, cal, grids, loo, vm_work


def build_reference_inspector(extractor, dataset, cfg: PipelineConfig, segmenter=None):
    """Inspector from the MSD good/ images (for evaluating on MSD)."""
    q = cfg.normal_jpeg_quality
    if q == "auto":
        q = dataset.estimate_defect_jpeg_quality()
    normals = [dataset.load_image(s, q) for s in dataset.normal_samples()]
    bank, cal, grids, loo, vms = build_bank_and_calibration(extractor, normals, cfg)
    cal.details["normal_jpeg_quality_used"] = q
    bank.meta["normal_jpeg_quality_used"] = q
    insp = DefectInspector(extractor, bank, cal, cfg, segmenter)
    insp.normal_grids, insp.normal_loo_heatmaps = grids, loo
    insp.normal_sizes = [work_size(im.width, im.height, cfg.work_long_side) for im in normals]
    return insp


def build_few_shot_inspector(extractor, clean_photos: list[Image.Image], cfg: PipelineConfig,
                             segmenter=None, roi_method: str | None = "auto", roi_kwargs: dict | None = None,
                             augment_flip: bool = True):
    """Inspector for a live phone from >= 3 clean photos of it.

    Each photo goes through the same ROI step used at test time; finger/bezel
    patches are left out of the bank via the ROI validity mask. With
    augment_flip, horizontally flipped copies are added (sharing the source
    photo's id, so leave-one-out still excludes them).
    """
    if len(clean_photos) < 3:
        raise ValueError("Few-shot calibration needs at least 3 clean photos (5-10 recommended).")
    rois, images, masks, gids = [], [], [], []
    for i, photo in enumerate(clean_photos):
        if roi_method:
            from src.roi.screen_roi import detect_screen

            kw = {"margin_frac": cfg.border_margin_frac or 0.03, "out_long_side": cfg.long_side}
            kw.update(roi_kwargs or {})
            roi = detect_screen(photo, roi_method, segmenter, **kw)
            img, vm = roi.image, roi.valid_mask
            rois.append(roi)
        else:
            img, vm = photo.convert("RGB"), None
        images.append(img); masks.append(vm); gids.append(i)
        if augment_flip:
            images.append(ImageOps.mirror(img)); masks.append(None if vm is None else vm[:, ::-1].copy())
            gids.append(i)
    bank, cal, grids, loo, vms = build_bank_and_calibration(extractor, images, cfg, masks, gids)
    insp = DefectInspector(extractor, bank, cal, cfg, segmenter, roi_method)
    insp.reference_rois = rois
    return insp
