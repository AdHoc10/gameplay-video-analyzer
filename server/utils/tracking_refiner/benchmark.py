"""
One-time embedder speed benchmark.

When both ResNet-50 and OSNet are available, we don't know in advance which
will be faster on the host's actual CUDA / CPU setup — OSNet has fewer
parameters but a less optimized inference path, ResNet-50 has more parameters
but a battle-tested CUDNN path. So we measure once per Python process and
cache the winner.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .embedders import (
    Embedder,
    OSNetEmbedder,
    ResNet50Embedder,
    TORCHREID_AVAILABLE,
)

logger = logging.getLogger(__name__)

# Cache: (device_str) -> chosen embedder instance + timing tuple
_cache: Dict[str, Tuple[Embedder, float, float]] = {}


def _time_embedder(
    embedder: Embedder,
    frame: np.ndarray,
    bboxes: np.ndarray,
    warmup: int = 3,
    iters: int = 10,
) -> float:
    """Median wall-clock seconds per `embed_batch` call."""
    for _ in range(warmup):
        embedder.embed_batch(frame, bboxes)
    if torch.cuda.is_available() and embedder.device.type == "cuda":
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        embedder.embed_batch(frame, bboxes)
        if torch.cuda.is_available() and embedder.device.type == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def pick_faster_embedder(
    device: Optional[torch.device] = None,
    n_crops: int = 16,
    crop_w: int = 64,
    crop_h: int = 128,
) -> Embedder:
    """
    Build both embedders, time them on a synthetic batch, return the faster.

    Result is cached per device for the lifetime of the Python process.
    """
    device = device or torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    cache_key = str(device)
    if cache_key in _cache:
        return _cache[cache_key][0]

    if not TORCHREID_AVAILABLE:
        chosen = ResNet50Embedder(device=device)
        _cache[cache_key] = (chosen, 0.0, float("inf"))
        return chosen

    # Build a dummy frame big enough to contain `n_crops` boxes laid out in a
    # grid. Anything will do; embedders only need RGB uint8.
    grid_cols = 4
    grid_rows = (n_crops + grid_cols - 1) // grid_cols
    frame_w = grid_cols * crop_w
    frame_h = grid_rows * crop_h
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(frame_h, frame_w, 3), dtype=np.uint8)

    bboxes = []
    for i in range(n_crops):
        r, c = divmod(i, grid_cols)
        x1, y1 = c * crop_w, r * crop_h
        bboxes.append([x1, y1, x1 + crop_w, y1 + crop_h])
    bboxes = np.array(bboxes, dtype=np.float32)

    try:
        resnet = ResNet50Embedder(device=device)
        osnet  = OSNetEmbedder(device=device)
    except Exception as e:
        logger.info("OSNet build failed during benchmark (%s); using ResNet-50.", e)
        chosen = ResNet50Embedder(device=device)
        _cache[cache_key] = (chosen, 0.0, float("inf"))
        return chosen

    t_resnet = _time_embedder(resnet, frame, bboxes)
    t_osnet  = _time_embedder(osnet,  frame, bboxes)

    if t_osnet <= t_resnet:
        chosen, loser_t = osnet, t_resnet
        winner_t = t_osnet
    else:
        chosen, loser_t = resnet, t_osnet
        winner_t = t_resnet

    logger.info(
        "[embedder benchmark on %s] resnet50=%.4fs osnet=%.4fs -> chose %s",
        cache_key, t_resnet, t_osnet, chosen.name,
    )

    _cache[cache_key] = (chosen, winner_t, loser_t)
    return chosen
