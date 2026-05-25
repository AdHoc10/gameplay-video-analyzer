"""
Side-effect-free port of the perspective-calibration pipeline from
`perspective_shift/pixel_yard_calib.ipynb`.

The existing helpers in `perspective_shift/pixel2yard_calibration.py` have
`plt.show()` and `print()` baked into them — fine for the notebook,
unusable in a per-frame loop. This module inlines the algorithm without
those side effects.

Pipeline (matches the notebook exactly):

  1) HSV white-mask:        cv2.inRange(hsv, (0,0,170), (180,60,255))
  2) Horizontal morph open: kernel = (w//5, 1)
  3) Hough lines (probabilistic): rho=1, theta=pi/180, thr=60,
                                  minLineLength=w//4, maxLineGap=40
  4) Cluster near-horizontal segment midpoint-Ys (slope_thresh=8, gap=12)
  5) Möbius fit:            yards = (a·y + b) / (c·y + 1),
                             initial p0 = [0.05, 20.0, 0.005], maxfev=10000
  6) Width law:             W_px(y) = K · (y − y_horizon),
                             y_horizon = -1/c
  7) Build correspondences (left + right sideline pts per yard line) and
     compute homography via cv2.findHomography(RANSAC, thr=0.5).

We deliberately skip `cv2.warpPerspective` of the full frame — the
bird's-eye heatmap is a synthetic canvas, so we only need `H` to map
player centers, not the warped image.

If fewer than 3 yard lines are detected, `compute_calibration` returns
`None` — callers should treat this as a transient calibration miss and
either reuse the last good calibration (see `CalibrationCache`) or
render an empty field for that frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    """Output of one successful calibration pass.

    Coordinate convention (matches the notebook):
        H @ [x_px, y_px, 1] → (lat_yd · w, lon_yd · w, w)
        After perspective division:
            lat_yd ∈ [0, FIELD_WIDTH_YARDS] across the lateral axis
            lon_yd is longitudinal yards along the field
        y_horizon: pixel y of the vanishing point (typically negative,
                    i.e. above the frame's top edge)
    """

    H:           np.ndarray   # (3, 3) image_xy -> (lat_yd, lon_yd)
    popt:        np.ndarray   # (3,) [a, b, c] of the Möbius fit
    y_horizon:   float
    y_min_yards: float        # smallest yard value among detected lines (= near camera)
    y_max_yards: float        # largest  yard value among detected lines (= far  camera)


# ---------------------------------------------------------------------------
# Tunables — copied verbatim from pixel2yard_calibration.py for parity.
# ---------------------------------------------------------------------------

# NFL field width, in yards (used to scale the homography's destination side).
# The actual measured field width could differ (the user's footage measures
# ~65 yd); the homography produces lateral yards on a [0, FIELD_WIDTH_YARDS]
# axis purely by construction. Downstream code translates this to the
# heatmap canvas using BirdsEyeConfig.field_width_yds.
FIELD_WIDTH_YARDS: float = 160.0 / 3.0   # ≈ 53.33

# Line-detection thresholds (copied from pixel2yard_calibration.detect_horizontal_lines).
_HSV_LOW  = (0, 0, 170)
_HSV_HIGH = (180, 60, 255)
_HOUGH_THRESHOLD       = 60
_HOUGH_MIN_LEN_DIVISOR = 4         # minLineLength = w // 4
_HOUGH_MAX_GAP         = 40
_MORPH_KERNEL_DIVISOR  = 5         # kernel width = w // 5

# Cluster thresholds (copied from cluster_line_ys defaults).
_CLUSTER_SLOPE_THRESH = 8
_CLUSTER_GAP          = 12

# RANSAC reproj threshold (copied from notebook cell d002213b).
_RANSAC_REPROJ_THR    = 0.5

# Scan band width when measuring each yard line's pixel width (copied from
# notebook cell 883caf66).
_WIDTH_SCAN_BAND      = 5

# Minimum yard lines needed for a stable Möbius + homography fit.
_MIN_DETECTED_LINES   = 3


# ---------------------------------------------------------------------------
# Pure-function pipeline
# ---------------------------------------------------------------------------

def _white_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, _HSV_LOW, _HSV_HIGH)


def _detect_horizontal_segments(
    white_mask: np.ndarray,
    frame_w:    int,
) -> Optional[np.ndarray]:
    """Return raw Hough segments (or None if no segments found)."""
    kernel_h = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(1, frame_w // _MORPH_KERNEL_DIVISOR), 1)
    )
    white_horiz = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_h)
    return cv2.HoughLinesP(
        white_horiz,
        rho=1,
        theta=np.pi / 180,
        threshold=_HOUGH_THRESHOLD,
        minLineLength=max(1, frame_w // _HOUGH_MIN_LEN_DIVISOR),
        maxLineGap=_HOUGH_MAX_GAP,
    )


def _cluster_line_ys(
    segments: Optional[np.ndarray],
    slope_thresh: int = _CLUSTER_SLOPE_THRESH,
    cluster_gap:  int = _CLUSTER_GAP,
) -> List[float]:
    """Filter to near-horizontal segments, then merge close midpoint Ys."""
    if segments is None:
        return []
    ys: List[float] = []
    for seg in segments:
        x1, y1, x2, y2 = seg[0]
        if abs(y2 - y1) <= slope_thresh:
            ys.append((y1 + y2) / 2.0)
    ys.sort()

    clusters: List[float] = []
    group: List[float] = []
    for y in ys:
        if not group or y - group[-1] <= cluster_gap:
            group.append(y)
        else:
            clusters.append(float(np.median(group)))
            group = [y]
    if group:
        clusters.append(float(np.median(group)))
    return clusters


def _mobius(y_px: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return (a * y_px + b) / (c * y_px + 1.0)


def _fit_mobius(
    detected_ys: List[float],
    visible_yard_start: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the Möbius model. `visible_yard_start` is the yard value assumed
    for the BOTTOM-most detected line (i.e., the line nearest the camera).
    Returns (pixel_ys, yard_vals, popt)."""
    pixel_ys  = np.array(detected_ys, dtype=float)
    stop_line = visible_yard_start + 5 * len(pixel_ys)
    yard_vals = np.flip(
        np.arange(visible_yard_start, stop_line, 5, dtype=float)
    )
    popt, _ = curve_fit(
        _mobius, pixel_ys, yard_vals,
        p0=[0.05, 20.0, 0.005],
        maxfev=10_000,
    )
    return pixel_ys, yard_vals, popt


def _build_homography(
    white_mask:  np.ndarray,
    pixel_ys:    np.ndarray,
    yard_vals:   np.ndarray,
    y_horizon:   float,
    frame_h:     int,
) -> Optional[np.ndarray]:
    """Use the linear width law W_px(y) = K·(y − y_horizon) to infer
    sideline positions at each yard line, then RANSAC findHomography."""

    yard_data = []
    for y_i, yd_i in zip(pixel_ys, yard_vals):
        y  = int(round(y_i))
        y0 = max(0, y - _WIDTH_SCAN_BAND)
        y1 = min(frame_h - 1, y + _WIDTH_SCAN_BAND)
        col_active = np.any(white_mask[y0:y1 + 1, :] > 0, axis=0)
        cols = np.where(col_active)[0]
        if len(cols) < 50:
            continue
        xl, xr = int(cols[0]), int(cols[-1])
        yard_data.append({
            "y_px":   y_i, "yards":  yd_i,
            "x_left": xl,  "x_right": xr,
            "W_px":   xr - xl,
            "x_mid":  (xl + xr) / 2.0,
        })

    if len(yard_data) < _MIN_DETECTED_LINES:
        return None

    ys_rel = np.array([d["y_px"] - y_horizon for d in yard_data])
    ws_px  = np.array([d["W_px"]             for d in yard_data])
    denom  = float(np.dot(ys_rel, ys_rel))
    if denom == 0.0:
        return None
    K        = float(np.dot(ys_rel, ws_px) / denom)
    x_center = float(np.mean([d["x_mid"] for d in yard_data]))

    src_pts: List[List[float]] = []
    dst_pts: List[List[float]] = []
    for d in yard_data:
        W_fit  = K * (d["y_px"] - y_horizon)
        xl_fit = x_center - W_fit / 2.0
        xr_fit = x_center + W_fit / 2.0
        src_pts += [[xl_fit, d["y_px"]], [xr_fit, d["y_px"]]]
        dst_pts += [[0.0, d["yards"]], [FIELD_WIDTH_YARDS, d["yards"]]]

    H, _inliers = cv2.findHomography(
        np.array(src_pts, dtype=np.float32),
        np.array(dst_pts, dtype=np.float32),
        cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJ_THR,
    )
    if H is None:
        return None
    return H.astype(np.float64)


def compute_calibration(
    frame_bgr: np.ndarray,
    visible_yard_start: int = 5,
) -> Optional[Calibration]:
    """Run the full pipeline on one frame.

    Returns None on any failure mode (too few detected lines, degenerate
    fit). The caller should treat None as a transient miss.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    white_mask = _white_mask(frame_bgr)
    segments   = _detect_horizontal_segments(white_mask, w)
    line_ys    = _cluster_line_ys(segments)
    if len(line_ys) < _MIN_DETECTED_LINES:
        return None

    try:
        pixel_ys, yard_vals, popt = _fit_mobius(line_ys, visible_yard_start)
    except Exception:
        return None
    if popt[2] == 0.0:
        return None
    y_horizon = -1.0 / float(popt[2])

    H = _build_homography(white_mask, pixel_ys, yard_vals, y_horizon, h)
    if H is None:
        return None

    return Calibration(
        H=H,
        popt=popt,
        y_horizon=y_horizon,
        y_min_yards=float(yard_vals.min()),
        y_max_yards=float(yard_vals.max()),
    )


def pixel_to_field(
    x_px: float, y_px: float, calib: Calibration
) -> Tuple[float, float]:
    """Project an image pixel to (lat_yd, lon_yd) via the calibration's H."""
    p = calib.H @ np.array([float(x_px), float(y_px), 1.0])
    w = float(p[2])
    if w == 0.0:
        return float("nan"), float("nan")
    return float(p[0] / w), float(p[1] / w)


# ---------------------------------------------------------------------------
# CalibrationCache — supports "every_frame" and "cached" cadence modes.
# ---------------------------------------------------------------------------

class CalibrationCache:
    """Wraps `compute_calibration` with the cadence policy.

    Modes:
      * "every_frame" — recompute every call.
      * "cached"      — recompute when (frame_idx % cache_every_n_frames) == 0;
                         otherwise return the last good calibration.

    On any failed recompute, we fall back to the last good calibration (or
    None if we have never had one yet).
    """

    def __init__(
        self,
        mode: str = "cached",
        cache_every_n_frames: int = 5,
        visible_yard_start: int = 5,
    ):
        if mode not in ("every_frame", "cached"):
            raise ValueError(f"unknown calibration mode: {mode!r}")
        self.mode = mode
        self.cache_every_n_frames = max(1, int(cache_every_n_frames))
        self.visible_yard_start   = visible_yard_start
        self._last_good: Optional[Calibration] = None

    def reset(self) -> None:
        """Drop the cached calibration (call at the start of each play)."""
        self._last_good = None

    def get(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
    ) -> Optional[Calibration]:
        should_recompute = (
            self.mode == "every_frame"
            or (frame_idx % self.cache_every_n_frames == 0)
            or self._last_good is None
        )
        if should_recompute:
            fresh = compute_calibration(frame_bgr, self.visible_yard_start)
            if fresh is not None:
                self._last_good = fresh
        return self._last_good
