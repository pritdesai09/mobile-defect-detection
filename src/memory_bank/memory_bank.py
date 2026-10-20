"""
Normal-feature memory bank (PatchCore / SPADE-style k-NN anomaly scoring).

All patch embeddings from all normal reference images go into ONE bank.
A test patch's anomaly score is the mean cosine distance to its k nearest
bank vectors (k=1 is the PatchCore/SPADE nearest-neighbour score).

New in this version:
  * Each bank vector remembers which reference image it came from
    (`image_ids`). This enables leave-one-out scoring: a normal reference
    image can be scored against the bank *without its own patches*, which
    gives an unbiased distribution of "normal" scores for calibration.
    Augmented copies of the same image share one id so they are excluded too.
  * Patches can be excluded at build time (e.g. finger/bezel patches in a
    live few-shot reference photo) via `patch_valid`.
  * Chunked exact search: torch on GPU when available, NumPy otherwise.
  * Saves the extractor config alongside the vectors and refuses to score
    features produced by a different extractor configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from src.feature_extraction.dinov2_extractor import PatchGrid


@dataclass
class MemoryBank:
    k: int = 1
    chunk_size: int = 1024
    device: str | None = None  # None = auto (cuda if torch+GPU available, else numpy)
    meta: dict = field(default_factory=dict)

    _bank: np.ndarray = field(default_factory=lambda: np.empty((0, 0), np.float32), repr=False)
    _image_ids: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32), repr=False)
    _torch_bank: object | None = field(default=None, repr=False)
    _torch_ids: object | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ build
    @property
    def size(self) -> int:
        return int(self._bank.shape[0])

    @property
    def embed_dim(self) -> int:
        return int(self._bank.shape[1]) if self._bank.size else 0

    @property
    def n_images(self) -> int:
        return int(len(np.unique(self._image_ids))) if self._image_ids.size else 0

    @property
    def image_ids(self) -> np.ndarray:
        return self._image_ids

    def build(
        self,
        normal_grids: list[PatchGrid],
        patch_valid: list[np.ndarray | None] | None = None,
        group_ids: list[int] | None = None,
        meta: dict | None = None,
    ) -> "MemoryBank":
        """patch_valid[i]: optional (grid_h, grid_w) bool; False patches are left out.
        group_ids[i]: reference-image id for grid i (defaults to i). Give augmented
        copies of one photo the same id so leave-one-out excludes all of them.
        """
        if not normal_grids:
            raise ValueError("Need at least one normal PatchGrid to build a memory bank.")
        group_ids = list(range(len(normal_grids))) if group_ids is None else list(group_ids)
        feats, ids = [], []
        for i, g in enumerate(normal_grids):
            f = g.features.reshape(-1, g.embed_dim).astype(np.float32)
            if patch_valid is not None and patch_valid[i] is not None:
                f = f[np.asarray(patch_valid[i]).reshape(-1).astype(bool)]
            feats.append(f)
            ids.append(np.full(len(f), group_ids[i], np.int32))
        self._bank = np.ascontiguousarray(np.concatenate(feats))
        self._image_ids = np.concatenate(ids)
        if meta is not None:
            self.meta = dict(meta)
        self._prepare_backend()
        return self

    def _prepare_backend(self) -> None:
        self._torch_bank = self._torch_ids = None
        want = self.device
        try:
            import torch

            if want is None:
                want = "cuda" if torch.cuda.is_available() else "numpy"
            if want != "numpy":
                self._torch_bank = torch.from_numpy(self._bank).to(want)
                self._torch_ids = torch.from_numpy(self._image_ids.astype(np.int64)).to(want)
        except ImportError:
            pass

    # ------------------------------------------------------------------ query
    def check_compatible(self, extractor_config: dict | None) -> None:
        bank_cfg = self.meta.get("extractor")
        if bank_cfg and extractor_config and bank_cfg != extractor_config:
            raise ValueError(
                "Memory bank was built with a different extractor configuration.\n"
                f"  bank:      {bank_cfg}\n  extractor: {extractor_config}"
            )

    def knn_distance(self, queries: np.ndarray, exclude_image: int | None = None) -> np.ndarray:
        """(N, D) L2-normalized queries -> (N,) mean cosine distance to k nearest bank vectors."""
        if self.size == 0:
            raise RuntimeError("MemoryBank.build() must be called before scoring.")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        n_avail = self.size - (int((self._image_ids == exclude_image).sum()) if exclude_image is not None else 0)
        k = max(1, min(self.k, n_avail))
        if self._torch_bank is not None:
            return self._knn_torch(queries, k, exclude_image)
        return self._knn_numpy(queries, k, exclude_image)

    def _knn_numpy(self, q: np.ndarray, k: int, exclude_image: int | None) -> np.ndarray:
        excl = (self._image_ids == exclude_image) if exclude_image is not None else None
        out = np.empty(len(q), np.float32)
        for s in range(0, len(q), self.chunk_size):
            sims = q[s : s + self.chunk_size] @ self._bank.T
            if excl is not None:
                sims[:, excl] = -np.inf
            if k == 1:
                top = sims.max(axis=1)
            else:
                top = np.partition(sims, -k, axis=1)[:, -k:].mean(axis=1)
            out[s : s + self.chunk_size] = 1.0 - top
        return out

    def _knn_torch(self, q: np.ndarray, k: int, exclude_image: int | None) -> np.ndarray:
        import torch

        bank, ids = self._torch_bank, self._torch_ids
        qt = torch.from_numpy(q).to(bank.device)
        excl = (ids == exclude_image) if exclude_image is not None else None
        outs = []
        with torch.inference_mode():
            for s in range(0, len(qt), self.chunk_size):
                sims = qt[s : s + self.chunk_size] @ bank.T
                if excl is not None:
                    sims[:, excl] = float("-inf")
                top = sims.topk(k, dim=1).values.mean(dim=1)
                outs.append((1.0 - top).float().cpu())
        return torch.cat(outs).numpy()

    def score_grid(self, grid: PatchGrid, exclude_image: int | None = None) -> np.ndarray:
        """(grid_h, grid_w) float32 map of raw cosine-distance anomaly scores.

        These raw scores are directly comparable across images, which is what
        makes the calibrated normal/defective decision possible. Do not min-max
        normalize them before thresholding or evaluation.
        """
        h, w, d = grid.features.shape
        return self.knn_distance(grid.features.reshape(-1, d), exclude_image).reshape(h, w)

    # ------------------------------------------------------------------ utils
    def subset(self, image_ids: list[int]) -> "MemoryBank":
        """A new bank holding only the given reference images (Experiment E)."""
        keep = np.isin(self._image_ids, np.asarray(image_ids))
        nb = MemoryBank(k=self.k, chunk_size=self.chunk_size, device=self.device, meta=dict(self.meta))
        nb._bank = np.ascontiguousarray(self._bank[keep])
        nb._image_ids = self._image_ids[keep]
        nb._prepare_backend()
        return nb

    def with_k(self, k: int) -> "MemoryBank":
        nb = MemoryBank(k=k, chunk_size=self.chunk_size, device=self.device, meta=dict(self.meta))
        nb._bank, nb._image_ids = self._bank, self._image_ids
        nb._torch_bank, nb._torch_ids = self._torch_bank, self._torch_ids
        return nb

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, bank=self._bank, image_ids=self._image_ids, k=np.array(self.k), meta=np.array(json.dumps(self.meta))
        )

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "MemoryBank":
        data = np.load(path, allow_pickle=False)
        mb = cls(k=int(data["k"]), device=device, meta=json.loads(str(data["meta"])))
        mb._bank = np.ascontiguousarray(data["bank"].astype(np.float32))
        mb._image_ids = data["image_ids"].astype(np.int32)
        mb._prepare_backend()
        return mb
