"""
Single source of truth for pipeline settings.

Every results JSON written by this project embeds `PipelineConfig.to_dict()`,
so each reported number can be traced back to the exact settings that
produced it. Defaults below are *starting points*: Phase 4 of the notebook
chooses the feature settings on the validation split, not the test split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


@dataclass
class PipelineConfig:
    # --- Feature extraction (DINOv2) ---
    model_name: str = "dinov2_vits14"      # or "dinov2_vits14_reg" (register variant, cleaner patch maps)
    resize_mode: str = "aspect"            # "aspect" keeps 16:9 (fixes squash); "square" = original 518x518
    long_side: int = 1036                  # aspect mode: long side in px, multiple of 14 (-> 74x42 grid for 16:9)
    square_size: int = 518                 # square mode only
    layers: tuple[int, ...] | None = None  # None = last layer; e.g. (8, 11) concatenates intermediate layers

    # --- Memory bank ---
    k: int = 1                             # 1 = PatchCore/SPADE nearest-neighbour distance

    # --- Heatmap ---
    work_long_side: int = 960              # heatmaps, GT masks and SAM 2 all operate at this resolution
    smooth_patches: float = 0.5            # Gaussian sigma in *patch* units (scales with resolution)

    # --- Calibration (label-free, from normal images only) ---
    pixel_percentile: float = 99.5         # low (grow) threshold = this percentile of normal pixel scores
    image_rule: str = "max"                # "max" or "mean_std" over leave-one-out normal image scores
    image_margin: float = 0.0              # relative margin added to the image threshold
    image_k_std: float = 3.0               # used when image_rule == "mean_std"

    # --- Region extraction ---
    border_margin_frac: float = 0.0        # ignore this fraction of width/height at each border
    min_area_patches: float = 0.25         # drop components smaller than this many patch footprints
    max_regions: int = 10

    # --- SAM 2 prompting ---
    prompt_strategy: str = "box_pos_neg"   # "point" | "multi_point" | "box_point" | "box_pos_neg"
    n_pos: int = 3
    n_neg: int = 3
    box_pad_frac: float = 0.15
    min_containment: float = 0.6           # SAM mask must lie mostly inside the (dilated) heatmap region
    containment_dilate_patches: float = 0.5

    # --- Data ---
    # MSD normals are PNG but defect images are JPEG. "auto" estimates the defect images' JPEG
    # quality from their quantization tables (label-free) and round-trips the normals to match,
    # so compression artifacts don't read as anomalies. None disables; an int forces a quality.
    normal_jpeg_quality: int | str | None = "auto"
    seed: int = 42

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layers"] = list(self.layers) if self.layers else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        d = dict(d)
        if d.get("layers"):
            d["layers"] = tuple(d["layers"])
        return cls(**d)

    def with_(self, **kwargs) -> "PipelineConfig":
        return replace(self, **kwargs)

    def extractor_key(self) -> dict:
        """The subset of settings that must match between memory bank and test features."""
        return {
            "model_name": self.model_name,
            "resize_mode": self.resize_mode,
            "long_side": self.long_side,
            "square_size": self.square_size,
            "layers": list(self.layers) if self.layers else None,
        }


# The original project settings, kept so the baseline can be reproduced exactly.
BASELINE_CONFIG = PipelineConfig(
    model_name="dinov2_vits14",
    resize_mode="square",
    square_size=518,
    k=9,
    prompt_strategy="point",
    normal_jpeg_quality=None,
)

# Settings for live phone photos (ROI crop + bezel margin).
LIVE_CONFIG = PipelineConfig(border_margin_frac=0.03)
