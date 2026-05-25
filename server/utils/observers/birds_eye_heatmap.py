"""
Bird's-eye heatmap drawer.

Replaces the original-frame heatmap with a top-down field plan view:
green field background, players plotted as colored dots at their
projected field coordinates, ball carrier highlighted pink.

Field geometry (per user spec):
    field_length_yds = 140   (user-given)
    field_width_yds  = 65    (placeholder until measured-width lands)
    width_buffer_yds = 10    (added on each side so the canvas tolerates
                              dispute between assumed and actual width)

Two configurable toggles (both ship for A/B):
    PerspectiveConfig.mode = "every_frame" | "cached"
        How often to recompute the homography. "cached" recomputes every
        N frames and reuses the last good H in between (typically ~5×
        cheaper at the cost of slightly stale calibration during fast
        camera pans).
    BirdsEyeConfig.orientation = "portrait" | "landscape"
        Canvas orientation. The user runs both and picks visually.

Cost optimization:
    We deliberately skip `cv2.warpPerspective` of the full frame. We only
    need `H` to map player bbox bottom-centers onto a synthetic canvas
    we draw ourselves. This saves ~40ms/frame on a typical 1080p source
    and is the dominant per-frame win.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from config import RESULTS, RESULTS_DIR
from utils.perspective import (
    CalibrationCache,
    FIELD_WIDTH_YARDS,
    pixel_to_field,
)
from utils.tracking_observer import FrameContext, TrackingObserver


# ---------------------------------------------------------------------------
# Configs (kept local to this module — no global singleton, callers
# instantiate explicitly so A/B switching is a one-line edit).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerspectiveConfig:
    """Cadence policy for the homography recomputation."""

    mode: str = "cached"                   # "every_frame" | "cached"
    cache_every_n_frames: int = 5
    visible_yard_start: int = 5             # bottom-most detected line = 5 yd


@dataclass(frozen=True)
class BirdsEyeConfig:
    """Canvas + style knobs for the bird's-eye heatmap."""

    field_length_yds:    float = 140.0
    field_width_yds:     float = 65.0
    width_buffer_yds:    float = 10.0
    px_per_yd:           int   = 8

    orientation:         str   = "portrait"   # "portrait" | "landscape"

    # Colours are BGR because cv2 draws in BGR.
    bg_color_bgr:        Tuple[int, int, int] = (34, 139, 34)    # forest green
    yard_line_color:     Tuple[int, int, int] = (220, 220, 220)  # light gray
    yard_line_10s_color: Tuple[int, int, int] = (255, 255, 255)  # white at 10-yd marks
    attack_color_bgr:    Tuple[int, int, int] = (0, 0, 255)      # red
    defense_color_bgr:   Tuple[int, int, int] = (0, 255, 0)      # green
    bc_color_bgr:        Tuple[int, int, int] = (180, 105, 255)  # hot pink
    unknown_color_bgr:   Tuple[int, int, int] = (200, 200, 200)  # gray (no side tag)

    dot_radius_px:       int = 8
    outline_thickness:   int = 2


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

class BirdsEyeHeatmapObserver(TrackingObserver):
    """Renders a bird's-eye view of player positions and writes a video.

    Use via `Tracking.replace_heatmap_drawer(BirdsEyeHeatmapObserver(...))`.
    """

    def __init__(
        self,
        owner,
        cfg: BirdsEyeConfig = BirdsEyeConfig(),
        persp_cfg: PerspectiveConfig = PerspectiveConfig(),
    ):
        self.owner = owner
        self.cfg = cfg
        self.persp_cfg = persp_cfg

        self._calib_cache = CalibrationCache(
            mode=persp_cfg.mode,
            cache_every_n_frames=persp_cfg.cache_every_n_frames,
            visible_yard_start=persp_cfg.visible_yard_start,
        )

        # Canvas geometry derived from config — fixed for the session.
        self._canvas_w, self._canvas_h = self._compute_canvas_dims()
        # Pre-built field template — reused every frame to skip re-drawing
        # the green background and yard grid.
        self._field_template: np.ndarray = self._build_field_template()

        self._writer: Optional[cv2.VideoWriter] = None
        self._raw_path:   Optional[str] = None
        self._final_path: Optional[str] = None

    # ---- lifecycle --------------------------------------------------------

    def on_session_start(self, owner) -> None:
        self._raw_path   = str(RESULTS_DIR / RESULTS.heatmap_video_raw)
        self._final_path = str(RESULTS_DIR / RESULTS.heatmap_video)
        self._writer = cv2.VideoWriter(
            self._raw_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            owner.fps,
            (self._canvas_w, self._canvas_h),
        )
        # Expose paths back to owner so the main loop's H.264 conversion
        # step finds them (mirrors `_OriginalHeatmapDrawer`'s contract).
        owner.heatmap_video_raw  = self._raw_path
        owner.heatmap_video_path = self._final_path

    def on_play_start(self, play) -> None:
        # Calibration is per-play: camera angle can shift between plays.
        self._calib_cache.reset()

    def on_tracking_results(self, frame: FrameContext) -> None:
        canvas = self._field_template.copy()

        calib = self._calib_cache.get(frame.frame_bgr, frame.frame_idx)
        if calib is None:
            # No calibration yet — emit an empty field so we still produce
            # frames at the writer's expected cadence.
            frame.heatmap_frame = canvas
            return

        if frame.results is None or frame.results[0].boxes.id is None:
            frame.heatmap_frame = canvas
            return

        boxes = frame.results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids   = frame.results[0].boxes.id.cpu().numpy().astype(int)
        bc_tid = frame.play.notes.get("ballcarrier_tid")
        id_to_side = frame.play.id_to_side

        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            # Player's ground point = bottom-center of bbox (feet).
            px = (x1 + x2) / 2.0
            py = float(y2)
            lat_yd, lon_yd = pixel_to_field(px, py, calib)
            if not (np.isfinite(lat_yd) and np.isfinite(lon_yd)):
                continue

            cxcy = self._field_to_canvas(lat_yd, lon_yd)
            if cxcy is None:
                continue
            cx, cy = cxcy

            color = self._color_for(int(tid), bc_tid, id_to_side)
            # White outline ring + filled colored dot — makes defenders
            # legible against the green field.
            cv2.circle(
                canvas, (cx, cy),
                self.cfg.dot_radius_px + self.cfg.outline_thickness,
                (255, 255, 255), -1,
            )
            cv2.circle(canvas, (cx, cy), self.cfg.dot_radius_px, color, -1)

        frame.heatmap_frame = canvas

    def on_frame_drawn(self, frame: FrameContext) -> None:
        if self._writer is not None and frame.heatmap_frame is not None:
            self._writer.write(frame.heatmap_frame)

    def on_session_end(self, owner) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    # ---- canvas helpers ---------------------------------------------------

    def _compute_canvas_dims(self) -> Tuple[int, int]:
        """Return (width, height) for the heatmap canvas."""
        long_px = int(self.cfg.field_length_yds * self.cfg.px_per_yd)
        lat_px  = int(
            (self.cfg.field_width_yds + 2 * self.cfg.width_buffer_yds)
            * self.cfg.px_per_yd
        )
        if self.cfg.orientation == "portrait":
            return lat_px, long_px            # tall canvas
        if self.cfg.orientation == "landscape":
            return long_px, lat_px            # wide canvas
        raise ValueError(f"unknown orientation: {self.cfg.orientation!r}")

    def _build_field_template(self) -> np.ndarray:
        """Pre-render the green background + yard line grid once."""
        canvas = np.full(
            (self._canvas_h, self._canvas_w, 3),
            self.cfg.bg_color_bgr,
            dtype=np.uint8,
        )

        s   = self.cfg.px_per_yd
        L   = int(self.cfg.field_length_yds)            # 0..L yards
        max_lat = self.cfg.field_width_yds + 2 * self.cfg.width_buffer_yds

        # Horizontal grid lines every 5 yards along the longitudinal axis.
        for yd in range(0, L + 1, 5):
            colour = (
                self.cfg.yard_line_10s_color if (yd % 10 == 0)
                else self.cfg.yard_line_color
            )
            thickness = 2 if (yd % 10 == 0) else 1

            if self.cfg.orientation == "portrait":
                # Length runs along Y → horizontal line at constant Y.
                y = int(yd * s)
                cv2.line(canvas, (0, y), (self._canvas_w - 1, y),
                         colour, thickness)
            else:
                # Length runs along X → vertical line at constant X.
                x = int(yd * s)
                cv2.line(canvas, (x, 0), (x, self._canvas_h - 1),
                         colour, thickness)

        # Sideline markers — vertical (or horizontal) lines at the
        # *assumed* sideline positions, so the user can see whether the
        # 65 yd assumption matches reality.
        side_l_yd = self.cfg.width_buffer_yds                  # left sideline
        side_r_yd = self.cfg.width_buffer_yds + self.cfg.field_width_yds

        if self.cfg.orientation == "portrait":
            for lat_yd in (side_l_yd, side_r_yd):
                x = int(lat_yd * s)
                if 0 <= x < self._canvas_w:
                    cv2.line(canvas, (x, 0), (x, self._canvas_h - 1),
                             (255, 255, 255), 2)
        else:
            for lat_yd in (side_l_yd, side_r_yd):
                y = int(lat_yd * s)
                if 0 <= y < self._canvas_h:
                    cv2.line(canvas, (0, y), (self._canvas_w - 1, y),
                             (255, 255, 255), 2)

        return canvas

    def _field_to_canvas(
        self, lat_yd: float, lon_yd: float
    ) -> Optional[Tuple[int, int]]:
        """Map (lat_yd, lon_yd) from the homography output to canvas pixels.

        `lat_yd` comes out of the homography on the FIELD_WIDTH_YARDS scale
        (≈ 53.33). We rescale it to the user-given field_width_yds (65 by
        default) so the assumed field width controls the canvas mapping.
        The buffer is added on both sides.

        Returns None if the projected point falls outside the canvas.
        """
        s = self.cfg.px_per_yd

        # Rescale lat from [0, FIELD_WIDTH_YARDS] to [0, field_width_yds],
        # then offset by the buffer.
        lat_scaled = lat_yd * (self.cfg.field_width_yds / FIELD_WIDTH_YARDS)
        lat_with_buffer = self.cfg.width_buffer_yds + lat_scaled

        if self.cfg.orientation == "portrait":
            cx = int(round(lat_with_buffer * s))
            # Higher lon_yd is farther from the camera; place it at the top.
            cy = int(round((self.cfg.field_length_yds - lon_yd) * s))
        else:
            # Landscape: length on X, width on Y.
            cx = int(round((self.cfg.field_length_yds - lon_yd) * s))
            cy = int(round(lat_with_buffer * s))

        if not (0 <= cx < self._canvas_w and 0 <= cy < self._canvas_h):
            return None
        return cx, cy

    def _color_for(
        self,
        tid: int,
        bc_tid: Optional[int],
        id_to_side,
    ) -> Tuple[int, int, int]:
        if bc_tid is not None and tid == bc_tid:
            return self.cfg.bc_color_bgr
        side = id_to_side.get(tid)
        if side == "A":
            return self.cfg.attack_color_bgr
        if side == "D":
            return self.cfg.defense_color_bgr
        return self.cfg.unknown_color_bgr
