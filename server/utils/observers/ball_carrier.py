"""
Ball-carrier detection observer.

Runs the fine-tuned `best_ballCarrier_detection.pt` model every frame and
binds the highest-confidence detection to a tracker ID via IoU. The bound
tid is published on `play.notes["ballcarrier_tid"]` so downstream
observers (e.g. the bird's-eye heatmap drawer) can render the ball carrier
distinctly.

Persistence policy (per user choice):
    Last known carrier tid persists indefinitely until a new detection
    arrives. This includes across play boundaries — the model can be
    noisy mid-pass, and tags should not flicker.

Cost: one extra YOLO forward pass per frame. Worst-case ~10ms on a GPU.
If this turns out to be too expensive, the easiest knob is a
`run_every_n_frames` config flag (not implemented here; users currently
prefer max accuracy on this signal).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import BALLCARRIER, BallCarrierConfig
from utils.geometry import iou
from utils.tracking_observer import FrameContext, PlayContext, TrackingObserver


class BallCarrierObserver(TrackingObserver):
    """Per-frame ball-carrier tid resolver."""

    def __init__(
        self,
        owner,
        cfg: BallCarrierConfig = BALLCARRIER,
    ):
        self.owner = owner
        self.cfg = cfg
        self._last_known_tid: Optional[int] = None

    # ---- lifecycle --------------------------------------------------------

    def on_play_start(self, play: PlayContext) -> None:
        # Persistent across plays — bring the last known tid forward so the
        # heatmap renders a pink dot from frame 0 of the new play even
        # before the model fires.
        play.notes["ballcarrier_tid"] = self._last_known_tid

    def on_tracking_results(self, frame: FrameContext) -> None:
        if frame.results is None or frame.results[0].boxes.id is None:
            frame.play.notes["ballcarrier_tid"] = self._last_known_tid
            return

        bc_res = self.owner.ballCarrier_detection_model(
            frame.frame_bgr, verbose=False
        )[0]

        if bc_res.boxes is None or len(bc_res.boxes) == 0:
            frame.play.notes["ballcarrier_tid"] = self._last_known_tid
            return

        confs = bc_res.boxes.conf.cpu().numpy()
        best  = int(np.argmax(confs))
        if confs[best] < self.cfg.conf_threshold:
            frame.play.notes["ballcarrier_tid"] = self._last_known_tid
            return

        bc_xyxy   = bc_res.boxes.xyxy[best].cpu().numpy()
        trk_boxes = frame.results[0].boxes.xyxy.cpu().numpy()
        trk_ids   = frame.results[0].boxes.id.cpu().numpy().astype(int)

        best_iou, best_tid = 0.0, None
        for tb, tid in zip(trk_boxes, trk_ids):
            v = iou(bc_xyxy, tb)
            if v > best_iou:
                best_iou, best_tid = v, int(tid)

        if best_iou >= self.cfg.iou_match_threshold and best_tid is not None:
            self._last_known_tid = best_tid

        # Whether we matched or not, publish the latest known tid so
        # downstream observers always have a value to read.
        frame.play.notes["ballcarrier_tid"] = self._last_known_tid
