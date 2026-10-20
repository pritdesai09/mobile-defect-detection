"""
DINOv2 patch-feature extractor.

Turns an RGB image into a grid of L2-normalized patch embeddings.

Changes from the original version:
  * resize_mode="aspect" (default) keeps the image's aspect ratio. The old
    square 518x518 resize squashed 1920x1080 images non-uniformly, so each
    patch covered ~52x29 source pixels and thin scratches became sub-pixel.
    For 16:9 input, long_side=1036 gives a 74x42 patch grid.
  * Supports the register variants ("dinov2_vits14_reg" etc.), which produce
    cleaner patch maps with fewer spurious high-norm artifact tokens.
  * Optional intermediate-layer features (layers=(8, 11) concatenates them).
  * Batched extraction for same-size images (all MSD images are 1920x1080).

torch is imported lazily so the rest of the package imports without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 14  # all DINOv2 ViT-*/14 models


@dataclass
class PatchGrid:
    """features: (grid_h, grid_w, embed_dim) float32, L2-normalized along the last axis.
    input_size: (width, height) the image was resized to before DINOv2.
    source_size: (width, height) of the image that was passed in.
    """

    features: np.ndarray
    input_size: tuple[int, int]
    source_size: tuple[int, int] | None = None

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.features.shape[0], self.features.shape[1]

    @property
    def embed_dim(self) -> int:
        return self.features.shape[-1]

    def astype(self, dtype) -> "PatchGrid":
        return PatchGrid(self.features.astype(dtype), self.input_size, self.source_size)


def compute_input_size(
    width: int, height: int, resize_mode: str = "aspect", long_side: int = 1036, square_size: int = 518
) -> tuple[int, int]:
    """Returns the (width, height) fed to DINOv2; both are multiples of the patch size."""
    if resize_mode == "square":
        if square_size % PATCH_SIZE:
            raise ValueError(f"square_size must be a multiple of {PATCH_SIZE}, got {square_size}")
        return square_size, square_size
    if resize_mode != "aspect":
        raise ValueError(f"resize_mode must be 'aspect' or 'square', got {resize_mode!r}")
    if long_side % PATCH_SIZE:
        raise ValueError(f"long_side must be a multiple of {PATCH_SIZE}, got {long_side}")
    scale = long_side / max(width, height)
    tw = max(PATCH_SIZE, int(round(width * scale / PATCH_SIZE)) * PATCH_SIZE)
    th = max(PATCH_SIZE, int(round(height * scale / PATCH_SIZE)) * PATCH_SIZE)
    return tw, th


def to_normalized_array(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """PIL -> (3, H, W) float32 array, resized and ImageNet-normalized."""
    arr = np.asarray(image.convert("RGB").resize(size, Image.BICUBIC), dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return arr.transpose(2, 0, 1).copy()


class DinoV2Extractor:
    """Extracts per-patch DINOv2 embeddings.

    model_name: "dinov2_vits14" (384-d), "dinov2_vitb14" (768-d), "dinov2_vitl14",
        or any of these with a "_reg" suffix for the register variants.
    resize_mode / long_side / square_size: see compute_input_size().
    layers: None for the final normalized patch tokens, or a tuple of block
        indices (e.g. (8, 11)) whose normalized outputs are concatenated.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        resize_mode: str = "aspect",
        long_side: int = 1036,
        square_size: int = 518,
        layers: tuple[int, ...] | None = None,
        device: str | None = None,
        batch_size: int = 8,
        image_size: int | None = None,  # backwards compatibility with the old signature
    ):
        import torch

        if image_size is not None:  # old call style DinoV2Extractor(image_size=518) meant square mode
            resize_mode, square_size = "square", image_size
        compute_input_size(1920, 1080, resize_mode, long_side, square_size)  # validates settings

        self.model_name = model_name
        self.resize_mode = resize_mode
        self.long_side = long_side
        self.square_size = square_size
        self.layers = tuple(layers) if layers else None
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Downloads + caches weights from Meta's hub on first use (needs internet).
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)

    @classmethod
    def from_config(cls, cfg, **kwargs) -> "DinoV2Extractor":
        return cls(cfg.model_name, cfg.resize_mode, cfg.long_side, cfg.square_size, cfg.layers, **kwargs)

    def config(self) -> dict:
        return {
            "model_name": self.model_name,
            "resize_mode": self.resize_mode,
            "long_side": self.long_side,
            "square_size": self.square_size,
            "layers": list(self.layers) if self.layers else None,
        }

    def input_size_for(self, image: Image.Image) -> tuple[int, int]:
        return compute_input_size(image.width, image.height, self.resize_mode, self.long_side, self.square_size)

    def _forward(self, batch: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(batch).to(self.device)
        with torch.inference_mode():
            if self.layers is None:
                tokens = self.model.forward_features(x)["x_norm_patchtokens"]  # CLS + registers excluded
            else:
                outs = self.model.get_intermediate_layers(x, n=list(self.layers), reshape=False, norm=True)
                tokens = torch.cat(list(outs), dim=-1)
            tokens = F.normalize(tokens.float(), dim=-1)
        gh, gw = grid_hw
        return tokens.reshape(tokens.shape[0], gh, gw, tokens.shape[-1]).cpu().numpy().astype(np.float32)

    def extract_patch_grid(self, image: Image.Image) -> PatchGrid:
        return self.extract_batch([image])[0]

    def extract_batch(self, images: list[Image.Image]) -> list[PatchGrid]:
        """Extracts grids for many images, batching together images of equal input size."""
        results: list[PatchGrid | None] = [None] * len(images)
        by_size: dict[tuple[int, int], list[int]] = {}
        for i, img in enumerate(images):
            by_size.setdefault(self.input_size_for(img), []).append(i)

        for size, idxs in by_size.items():
            grid_hw = (size[1] // PATCH_SIZE, size[0] // PATCH_SIZE)
            for start in range(0, len(idxs), self.batch_size):
                chunk = idxs[start : start + self.batch_size]
                batch = np.stack([to_normalized_array(images[i], size) for i in chunk])
                feats = self._forward(batch, grid_hw)
                for j, i in enumerate(chunk):
                    results[i] = PatchGrid(feats[j], size, images[i].size)
        return results  # type: ignore[return-value]
