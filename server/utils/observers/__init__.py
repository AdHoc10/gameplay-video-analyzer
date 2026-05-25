"""
Plug-in tracking observers.

Each observer is a self-contained TrackingObserver that can be added to a
`Tracking` instance via `register_observer()` or, for heatmap drawers,
`replace_heatmap_drawer()`.

Public surface:
    BallCarrierObserver         — per-frame ball-carrier tid resolver
    BirdsEyeHeatmapObserver     — bird's-eye field heatmap drawer
    BirdsEyeConfig              — canvas/style knobs for the bird's-eye drawer
    PerspectiveConfig           — calibration cadence policy
"""

from .ball_carrier import BallCarrierObserver
from .birds_eye_heatmap import (
    BirdsEyeConfig,
    BirdsEyeHeatmapObserver,
    PerspectiveConfig,
)

__all__ = [
    "BallCarrierObserver",
    "BirdsEyeHeatmapObserver",
    "BirdsEyeConfig",
    "PerspectiveConfig",
]
