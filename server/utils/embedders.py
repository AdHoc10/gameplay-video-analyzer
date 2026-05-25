"""
Player appearance embedders. Shared module — used by the tracking refiner
and available for any other feature that needs CNN appearance features
(e.g., team_learning_and_detection.py if it's reactivated).

Two interchangeable implementations behind a common `Embedder` interface:

  * `ResNet50Embedder` — ImageNet-pretrained ResNet-50 with the classification
    head replaced by Identity. Produces 2048-D embeddings. Always available
    (torch + torchvision are required by the project anyway).

  * `OSNetEmbedder` — torchreid's OSNet, purpose-built for person re-id.
    Produces 512-D embeddings. Optional: requires `torchreid` to be installed.
    If unavailable, `build_default_embedder` silently falls back to ResNet-50.

Both produce L2-normalized float32 ndarrays so downstream cosine similarity
is just a dot product.

The ResNet50Embedder applies `/255` scaling + ImageNet mean/std normalization,
which is the canonical pretrained-ResNet preprocessing. The earlier inline
embedder in team_learning_and_detection.py omitted these steps — fine for
GMM clustering, off-spec for cosine-similarity re-id.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import cv2
import numpy as np
import torch
from torchvision.models.resnet import resnet50, ResNet50_Weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class Embedder(ABC):
    """Common contract every embedder honours."""

    name: str
    dim: int
    device: torch.device

    @abstractmethod
    def embed_batch(
        self,
        frame_rgb: np.ndarray,
        bboxes_xyxy: np.ndarray,
    ) -> np.ndarray:
        """
        Compute one L2-normalized embedding per bbox crop.

        Args:
            frame_rgb:   HxWx3 uint8 RGB frame.
            bboxes_xyxy: (N, 4) int/float array of [x1, y1, x2, y2].

        Returns:
            (N, self.dim) float32, every row L2-normalized.
            Empty inputs return shape (0, self.dim).
        """
        ...

    # ---- shared helpers ----------------------------------------------------

    def _extract_crops(
        self,
        frame_rgb: np.ndarray,
        bboxes_xyxy: np.ndarray,
    ) -> List[np.ndarray]:
        """Clip bboxes to frame bounds and return per-crop ndarrays (RGB)."""
        h, w = frame_rgb.shape[:2]
        crops: List[np.ndarray] = []
        for box in bboxes_xyxy:
            x1, y1, x2, y2 = box[:4]
            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(0, min(w,     x2)))
            y2 = int(max(0, min(h,     y2)))
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                # Degenerate box — emit a tiny placeholder so the caller can
                # keep its index alignment. embed_batch will replace its row
                # with a zero vector below.
                crops.append(None)  # type: ignore[arg-type]
                continue
            crops.append(frame_rgb[y1:y2, x1:x2])
        return crops

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return (x / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# ResNet-50
# ---------------------------------------------------------------------------

# Standard ImageNet normalization. The original team_learning embedder skipped
# this — it worked for GMM clustering but degrades cosine-similarity re-id.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ResNet50Embedder(Embedder):
    """ImageNet-pretrained ResNet-50 with fc replaced by Identity."""

    name = "resnet50"
    dim = 2048

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(self.device)

    def embed_batch(
        self,
        frame_rgb: np.ndarray,
        bboxes_xyxy: np.ndarray,
    ) -> np.ndarray:
        if bboxes_xyxy is None or len(bboxes_xyxy) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        crops = self._extract_crops(frame_rgb, np.asarray(bboxes_xyxy))

        # Build a batch tensor; record which indices are valid.
        tensors = []
        valid_mask = np.zeros(len(crops), dtype=bool)
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            t = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
            t = t.astype(np.float32) / 255.0
            t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
            t = np.moveaxis(t, -1, 0)  # HWC -> CHW
            tensors.append(t)
            valid_mask[i] = True

        out = np.zeros((len(crops), self.dim), dtype=np.float32)
        if not tensors:
            return out

        batch = torch.from_numpy(np.stack(tensors, axis=0)).to(self.device)
        with torch.no_grad():
            feats = self.model(batch).cpu().numpy().astype(np.float32)

        out[valid_mask] = feats
        # Only normalize the valid rows; zero rows stay zero.
        if valid_mask.any():
            out[valid_mask] = self._l2_normalize(out[valid_mask])
        return out


# ---------------------------------------------------------------------------
# OSNet (optional)
# ---------------------------------------------------------------------------

try:
    from torchreid.utils import FeatureExtractor as _TorchReIDFeatureExtractor
    TORCHREID_AVAILABLE = True
except Exception:  # pragma: no cover - import-time only
    _TorchReIDFeatureExtractor = None  # type: ignore[assignment]
    TORCHREID_AVAILABLE = False


class OSNetEmbedder(Embedder):
    """torchreid OSNet (osnet_x1_0 by default). 512-D embeddings."""

    name = "osnet_x1_0"
    dim = 512

    def __init__(
        self,
        device: Optional[torch.device] = None,
        model_name: str = "osnet_x1_0",
    ):
        if not TORCHREID_AVAILABLE:
            raise RuntimeError(
                "torchreid is not installed. Install with "
                "`pip install torchreid` or use ResNet50Embedder."
            )
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        # `FeatureExtractor` accepts a device string like 'cuda:0' / 'cpu'.
        self._extractor = _TorchReIDFeatureExtractor(  # type: ignore[misc]
            model_name=model_name,
            device=str(self.device),
        )
        self.name = model_name

    def embed_batch(
        self,
        frame_rgb: np.ndarray,
        bboxes_xyxy: np.ndarray,
    ) -> np.ndarray:
        if bboxes_xyxy is None or len(bboxes_xyxy) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        crops = self._extract_crops(frame_rgb, np.asarray(bboxes_xyxy))

        valid_indices = [i for i, c in enumerate(crops) if c is not None and c.size]
        out = np.zeros((len(crops), self.dim), dtype=np.float32)
        if not valid_indices:
            return out

        valid_crops = [crops[i] for i in valid_indices]
        # FeatureExtractor accepts a list of ndarrays (BGR by convention) or
        # PIL Images. We pass BGR per torchreid's documented expectation.
        valid_bgr = [cv2.cvtColor(c, cv2.COLOR_RGB2BGR) for c in valid_crops]

        with torch.no_grad():
            feats = self._extractor(valid_bgr)
        feats_np = feats.detach().cpu().numpy().astype(np.float32)

        for slot, vec in zip(valid_indices, feats_np):
            out[slot] = vec
        # Normalize only the rows that were filled.
        valid_mask = np.zeros(len(crops), dtype=bool)
        valid_mask[valid_indices] = True
        out[valid_mask] = self._l2_normalize(out[valid_mask])
        return out


# ---------------------------------------------------------------------------
# Default builder
# ---------------------------------------------------------------------------

def build_default_embedder(
    device: Optional[torch.device] = None,
    prefer: Optional[str] = None,
) -> Embedder:
    """
    Build the embedder the refiner should use by default.

    Selection rule:
      * If `prefer` is given and matches a known name ("resnet50" or "osnet"),
        return that explicitly. ResNet-50 is always available; OSNet falls
        back to ResNet-50 if torchreid is missing.
      * Otherwise, if torchreid is available, run a tiny per-device benchmark
        (see benchmark.pick_faster_embedder) and return the faster one.
      * Otherwise, return ResNet-50.

    The benchmark result is cached per process so importing/instantiating
    the refiner multiple times is cheap.
    """
    if prefer is not None:
        prefer = prefer.lower()
        if prefer.startswith("resnet"):
            return ResNet50Embedder(device=device)
        if prefer.startswith("osnet"):
            if not TORCHREID_AVAILABLE:
                logger.info(
                    "OSNet requested but torchreid not installed; "
                    "falling back to ResNet-50."
                )
                return ResNet50Embedder(device=device)
            return OSNetEmbedder(device=device)

    if not TORCHREID_AVAILABLE:
        logger.info("torchreid not installed; defaulting to ResNet-50 embedder.")
        return ResNet50Embedder(device=device)

    # Both are available — let the benchmark decide.
    from .benchmark import pick_faster_embedder
    return pick_faster_embedder(device=device)
