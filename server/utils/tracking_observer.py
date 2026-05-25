"""
Observer contract for the per-play tracking loop.

`Tracking.track_in_each_play` emits events at named points; any number of
`TrackingObserver` subclasses can register and react. This is the seam that
keeps "future feature" code (refiner, ball-carrier ID, perspective shift,
custom metrics) out of the core loop.

Lifecycle for one tracking session:
    on_session_start(owner)              # once, before any play
    for each play:
        on_play_start(play_ctx)
        for each frame:
            on_tracking_results(frame_ctx)   # mutate id_to_side here if needed
            on_frame_drawn(frame_ctx)        # frame has annotated_frame/heatmap_frame
        on_play_end(play_ctx)
    on_session_end(owner)                # once, after the last play

All observer methods are no-ops by default; subclass and override only the
ones you care about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------

@dataclass
class PlayContext:
    """State scoped to one play. Mutable; observers may extend with notes."""

    play_index:        int
    start_time_frame:  int
    moves:             List[Tuple[str, int, Optional[int]]]
    id_to_side:        Dict[int, str]            = field(default_factory=dict)
    id_to_initial_y:   Dict[int, float]          = field(default_factory=dict)
    play_source_start: Optional[int]             = None
    play_source_end:   Optional[int]             = None
    play_output_start: int                       = 0
    notes:             Dict[str, Any]            = field(default_factory=dict)


@dataclass
class FrameContext:
    """State scoped to one frame within a play."""

    play:              PlayContext
    frame_idx:          int
    frame_bgr:          np.ndarray                 # original frame
    annotated_frame:    Optional[np.ndarray] = None
    heatmap_frame:      Optional[np.ndarray] = None
    results:            Any = None                 # ultralytics track() result
    is_snap_frame:      bool = False

    # Lazy RGB conversion — observers that want it call `frame_ctx.frame_rgb`.
    _frame_rgb_cache:   Optional[np.ndarray] = None

    @property
    def frame_rgb(self) -> np.ndarray:
        import cv2
        if self._frame_rgb_cache is None:
            self._frame_rgb_cache = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)
        return self._frame_rgb_cache


# ---------------------------------------------------------------------------
# Observer base
# ---------------------------------------------------------------------------

class TrackingObserver:
    """Subclass and override the hooks you care about."""

    def on_session_start(self, owner) -> None:
        """Called once before the first play, after `initialize_video_handling`.

        Observers that own external resources (VideoWriters, model handles,
        sockets) should initialize them here using `owner.fps`, `owner.w`,
        `owner.h`. Released later in `on_session_end`.
        """
        ...

    def on_play_start(self, play: PlayContext) -> None: ...

    def on_tracking_results(self, frame: FrameContext) -> None:
        """Called after track(persist=True) but BEFORE drawing.

        Use this to mutate `frame.play.id_to_side` (re-id, swap-fix, etc.)
        so subsequent drawing/event-collection sees the corrected tags."""
        ...

    def on_frame_drawn(self, frame: FrameContext) -> None:
        """Called after the default drawer has populated `annotated_frame`
        and `heatmap_frame`, before they are written to disk. Useful for
        overlays or custom heatmaps."""
        ...

    def on_play_end(self, play: PlayContext) -> None: ...

    def on_session_end(self, owner) -> None:
        """Called once after the last play, before the H.264 conversion step.

        Release VideoWriters and any other resources allocated in
        `on_session_start`.
        """
        ...
