"""
Embedding-based tracking refiner.

Drop-in subclass of `utils.tracking.Tracking` that augments ByteTrack with a
per-player appearance gallery. Use:

    from utils.tracking_refiner import RefinedTracking
    tracker = RefinedTracking(csv_name=csv_name, frame_handler=frame_handler)
    tracker.track_in_each_play()

The refiner has no side effects when not invoked: existing `Tracking` users
are not affected.

NOTE on imports: `RefinedTracking` transitively pulls in `utils.tracking` and
its dependency chain (yt_dlp, ultralytics, ...). The lightweight pieces
(gallery, refiner config, embedders) can be used in isolation, e.g. for
unit tests. We therefore expose `RefinedTracking` lazily via PEP 562 so a
plain `from utils.tracking_refiner import TrackGallery` works in a minimal
environment.
"""

from .embedders import (
    Embedder,
    OSNetEmbedder,
    ResNet50Embedder,
    TORCHREID_AVAILABLE,
    build_default_embedder,
)
from .gallery import TrackGallery
from .refiner import RefineOutput, RefinerConfig, TrackingRefiner

__all__ = [
    "Embedder",
    "OSNetEmbedder",
    "ResNet50Embedder",
    "TORCHREID_AVAILABLE",
    "build_default_embedder",
    "TrackGallery",
    "TrackingRefiner",
    "RefinerConfig",
    "RefineOutput",
    "RefinedTracking",
]


def __getattr__(name):
    if name == "RefinedTracking":
        from .refined_tracking import RefinedTracking
        return RefinedTracking
    raise AttributeError(f"module 'utils.tracking_refiner' has no attribute {name!r}")
