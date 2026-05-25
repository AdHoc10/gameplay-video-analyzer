"""
Drop-in subclass of `utils.tracking.Tracking` that runs an embedding-based
refiner alongside ByteTrack.

Now that `Tracking` exposes an observer hook API, this is a tiny class:
register one `_RefinerObserver` and you're done — no duplicated loop body.

Usage:
    from utils.tracking_refiner import RefinedTracking
    tracker = RefinedTracking(csv_name=csv_name, frame_handler=frame_handler)
    tracker.track_in_each_play()
"""

from __future__ import annotations

import logging
from typing import Optional

from utils.tracking import Tracking
from utils.tracking_observer import FrameContext, PlayContext, TrackingObserver

from .embedders import Embedder, build_default_embedder
from .refiner import RefinerConfig, TrackingRefiner

logger = logging.getLogger(__name__)


class _RefinerObserver(TrackingObserver):
    """Bridges `TrackingRefiner` into the Tracking observer pipeline."""

    def __init__(self, refiner: TrackingRefiner):
        self.refiner = refiner

    def on_play_start(self, play: PlayContext) -> None:
        # Fresh appearance gallery for every play.
        self.refiner.reset()

    def on_tracking_results(self, frame: FrameContext) -> None:
        # Runs BEFORE the default drawer (insertion order in Tracking),
        # so id_to_side updates are reflected in the rendered frame.
        if frame.results is None or frame.results[0].boxes.id is None:
            return
        self.refiner.refine(
            frame_rgb  = frame.frame_rgb,
            boxes_xyxy = frame.results[0].boxes.xyxy.cpu().numpy(),
            tids       = frame.results[0].boxes.id.cpu().numpy().astype(int),
            id_to_side = frame.play.id_to_side,
            frame_idx  = frame.frame_idx,
        )


class RefinedTracking(Tracking):
    """`Tracking` + embedding-based re-id / swap-fix refinement."""

    def __init__(
        self,
        csv_name,
        frame_handler,
        embedder: Optional[Embedder] = None,
        refiner_cfg: Optional[RefinerConfig] = None,
    ):
        super().__init__(csv_name=csv_name, frame_handler=frame_handler)
        self.embedder = embedder or build_default_embedder()
        self.refiner_cfg = refiner_cfg or RefinerConfig()
        self.refiner = TrackingRefiner(self.embedder, self.refiner_cfg)
        self.register_observer(_RefinerObserver(self.refiner))
        logger.info(
            "[RefinedTracking] using embedder=%s dim=%d device=%s",
            self.embedder.name, self.embedder.dim, str(self.embedder.device),
        )
