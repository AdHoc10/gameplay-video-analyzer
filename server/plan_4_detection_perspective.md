# Plan — Ball Carrier Detection + Perspective-Shifted Bird's-Eye Heatmap

## Context

Two server-side features to add, both as pluggable observers on top of the
existing `Tracking` observer pipeline:

1. **Ball carrier detection.** The model `best_ballCarrier_detection.pt` is
   already loaded in `Tracking.__init__` ([utils/tracking.py:199](server/utils/tracking.py)) but never used.
   It needs to run **every frame** (passes change possession mid-play). The
   ball-carrier `tid` then drives a pink-dot rendering on the heatmap.

2. **Perspective shift + bird's-eye heatmap.** The current heatmap is bbox
   centers + intra-team lines drawn on a black canvas in *original-frame*
   coordinates. The user wants a true bird's-eye field plan:
   - Field colored green, attackers red, defenders green (with a white outline
     so they read against the field), ball carrier pink.
   - Player positions are projected via the homography computed from yard-line
     detection (the algorithm in `perspective_shift/pixel_yard_calib.ipynb`,
     using helpers from `perspective_shift/pixel2yard_calibration.py`).
   - Canvas dimensions reflect the actual field aspect ratio: length = 140 yd,
     width = 65 yd (with a ±10 yd buffer, so the canvas holds up to 75 yd
     laterally until real-time field-width measurement lands).
   - **Orientation toggle**: portrait (length vertical) or landscape — both
     ship, switchable via one config field. *(User wants to try both.)*
   - **Calibration cadence toggle**: "every_frame" (most accurate) or "cached"
     (recompute every K frames + reuse last H — ~K× cheaper). Both ship,
     switchable via one config field. *(User wants to try both.)*
   - **Cost optimization**: we do NOT `cv2.warpPerspective` the full frame.
     We only need `H` to project player centers onto a synthetic green
     canvas. Skipping the warp eliminates the bulk of the per-frame cost.
   - **Ball-carrier gap handling**: persist last known carrier indefinitely
     until a new detection arrives. *(User choice.)*
   - OCR-based yard labeling and the existing field heatmap from
     `final_pixel2yard_calibration.py` are out of scope (user said both
     perform poorly).

Both features ship as `TrackingObserver` subclasses so they integrate with
zero changes to the main tracking loop's lifecycle.

## Files to create

```
server/utils/perspective.py                   # side-effect-free homography pipeline
server/utils/observers/__init__.py            # package surface
server/utils/observers/ball_carrier.py        # BallCarrierObserver
server/utils/observers/birds_eye_heatmap.py   # BirdsEyeHeatmapObserver + BirdsEyeConfig
```

The existing helpers in `perspective_shift/pixel2yard_calibration.py` have
`plt.show()` calls and `print()` statements baked into them — they're meant
for notebook exploration, not a per-frame loop. `perspective.py` ports the
algorithm without those side effects, but keeps the math identical so the
notebook's calibration numbers reproduce.

## Module 1 — `server/utils/perspective.py`

A small, pure-Python module. No I/O, no plotting.

```python
@dataclass(frozen=True)
class Calibration:
    H:           np.ndarray   # (3,3) image_xy -> (lat_yd, lon_yd)
    popt:        np.ndarray   # (3,) Möbius [a, b, c]
    y_horizon:   float        # = -1/c
    y_min_yards: float        # near-camera (bottom) yard
    y_max_yards: float        # far-camera  (top)    yard

def compute_calibration(
    frame_bgr: np.ndarray,
    visible_yard_start: int = 5,  # assume bottom-most detected line = 5 yd
) -> Optional[Calibration]:
    """
    Steps from pixel_yard_calib.ipynb, inlined and silenced:
      1) HSV white mask
      2) horizontal morphological open + HoughLinesP
      3) cluster y-positions of segments
      4) Möbius fit yards = (a·y + b)/(c·y + 1)
      5) width law K·(y - y_horizon); RANSAC findHomography
    Returns None if fewer than 3 yard lines detected (the algorithm needs ≥3
    for a stable fit).
    """

def pixel_to_field(
    x_px: float, y_px: float, calib: Calibration
) -> Tuple[float, float]:
    """H @ [x, y, 1], perspective division → (lat_yd, lon_yd)."""

class CalibrationCache:
    """Wraps compute_calibration with the every-K-frames-with-fallback policy.

    On `get(frame_bgr, frame_idx)`:
      * mode == "every_frame"  → recompute every call
      * mode == "cached"       → recompute when (frame_idx % cache_every_n) == 0;
                                 otherwise return last good calibration.
      * On a failed recompute, return the last good calibration (or None if
        nothing yet).
    """
```

`compute_calibration` is a port of the algorithm with `plt.show()` and `print`
stripped. It re-uses the same constants from the notebook (HSV thresholds, morph
kernel sizes, Hough params, slope/cluster thresholds), so the notebook stays
the canonical reference.

## Module 2 — `server/utils/observers/ball_carrier.py`

```python
class BallCarrierObserver(TrackingObserver):
    def __init__(self, owner: Tracking, cfg: BallCarrierConfig = ...):
        self.owner = owner
        self.cfg = cfg
        # Cross-frame, cross-play state: persistent last-known tid.
        self._last_known_tid: Optional[int] = None

    def on_play_start(self, play):
        # Per user choice: do NOT reset across plays.
        # The last-known tid persists until a new detection arrives.
        play.notes["ballcarrier_tid"] = self._last_known_tid

    def on_tracking_results(self, frame):
        # Runs BEFORE the heatmap drawer (observer ordering invariant).
        if frame.results is None or frame.results[0].boxes.id is None:
            return
        bc_res = self.owner.ballCarrier_detection_model(
            frame.frame_bgr, verbose=False
        )[0]
        if bc_res.boxes is None or len(bc_res.boxes) == 0:
            # No detection this frame — keep last_known.
            frame.play.notes["ballcarrier_tid"] = self._last_known_tid
            return
        # Highest-confidence ball carrier bbox.
        confs = bc_res.boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        if confs[best] < self.cfg.conf_threshold:
            frame.play.notes["ballcarrier_tid"] = self._last_known_tid
            return
        bc_xyxy = bc_res.boxes.xyxy[best].cpu().numpy()
        # IoU-match to tracker bboxes.
        trk_boxes = frame.results[0].boxes.xyxy.cpu().numpy()
        trk_ids   = frame.results[0].boxes.id.cpu().numpy().astype(int)
        best_iou, best_tid = 0.0, None
        for tb, tid in zip(trk_boxes, trk_ids):
            v = iou(bc_xyxy, tb)
            if v > best_iou:
                best_iou, best_tid = v, int(tid)
        if best_iou >= self.cfg.iou_match_threshold:
            self._last_known_tid = best_tid
            frame.play.notes["ballcarrier_tid"] = best_tid
        else:
            frame.play.notes["ballcarrier_tid"] = self._last_known_tid
```

`BallCarrierConfig` (added to `config.py`): `conf_threshold=0.4`,
`iou_match_threshold=0.3`.

## Module 3 — `server/utils/observers/birds_eye_heatmap.py`

```python
@dataclass(frozen=True)
class PerspectiveConfig:
    mode: str = "cached"                  # "every_frame" | "cached"
    cache_every_n_frames: int = 5
    visible_yard_start: int = 5            # bottom-most line assumed = 5 yd

@dataclass(frozen=True)
class BirdsEyeConfig:
    field_length_yds:  float = 140.0       # user-given
    field_width_yds:   float = 65.0        # user-given (placeholder)
    width_buffer_yds:  float = 10.0
    px_per_yd:         int   = 8
    orientation:       str   = "portrait"  # "portrait" | "landscape"
    bg_color_bgr:      Tuple[int,int,int] = (34, 139, 34)   # forest green
    yard_line_color:   Tuple[int,int,int] = (220, 220, 220)
    attack_color_bgr:  Tuple[int,int,int] = (0, 0, 255)     # red
    defense_color_bgr: Tuple[int,int,int] = (0, 255, 0)     # green
    bc_color_bgr:      Tuple[int,int,int] = (180, 105, 255) # pink
    dot_radius_px:     int   = 8
    outline_thickness: int   = 2


class BirdsEyeHeatmapObserver(TrackingObserver):
    """Owns its own VideoWriter for the bird's-eye heatmap output.

    Replaces the existing _OriginalHeatmapDrawer (see tracking.py change
    below). Register via `tracker.replace_heatmap_drawer(...)`."""

    def __init__(
        self,
        owner: Tracking,
        cfg: BirdsEyeConfig = BirdsEyeConfig(),
        persp_cfg: PerspectiveConfig = PerspectiveConfig(),
    ):
        self.owner = owner
        self.cfg = cfg
        self.persp_cfg = persp_cfg
        self._calib_cache = CalibrationCache(persp_cfg)
        self._writer: Optional[cv2.VideoWriter] = None
        self._canvas_w, self._canvas_h = self._compute_canvas_dims()

    def _compute_canvas_dims(self) -> Tuple[int, int]:
        long_px = int(self.cfg.field_length_yds * self.cfg.px_per_yd)
        lat_px  = int((self.cfg.field_width_yds + 2*self.cfg.width_buffer_yds)
                      * self.cfg.px_per_yd)
        if self.cfg.orientation == "portrait":
            return lat_px, long_px        # (w, h)
        return long_px, lat_px            # landscape: (w, h)

    def on_session_start(self, owner):
        # New hook — see tracking.py changes below.
        path = str(RESULTS_DIR / RESULTS.heatmap_video_raw)
        self._writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            owner.fps,
            (self._canvas_w, self._canvas_h),
        )

    def on_tracking_results(self, frame):
        calib = self._calib_cache.get(frame.frame_bgr, frame.frame_idx)
        canvas = self._build_field_canvas()           # green bg + yard grid
        if calib is None or frame.results is None or frame.results[0].boxes.id is None:
            frame.heatmap_frame = canvas
            return

        boxes = frame.results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids   = frame.results[0].boxes.id.cpu().numpy().astype(int)
        bc_tid = frame.play.notes.get("ballcarrier_tid")
        id_to_side = frame.play.id_to_side

        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            # Use bbox bottom-center as the player's ground point (feet).
            px, py = (x1 + x2) / 2.0, float(y2)
            lat, lon = pixel_to_field(px, py, calib)
            cx, cy = self._field_to_canvas(lat, lon)
            if cx is None:
                continue                   # off-canvas
            color = self._color_for(int(tid), bc_tid, id_to_side)
            cv2.circle(canvas, (cx, cy),
                       self.cfg.dot_radius_px + self.cfg.outline_thickness,
                       (255, 255, 255), -1)
            cv2.circle(canvas, (cx, cy), self.cfg.dot_radius_px, color, -1)

        frame.heatmap_frame = canvas

    def on_frame_drawn(self, frame):
        if self._writer is not None and frame.heatmap_frame is not None:
            self._writer.write(frame.heatmap_frame)

    def on_session_end(self, owner):
        if self._writer is not None:
            self._writer.release()
```

`_build_field_canvas` draws the green background + a 5-yard grid in light
gray + bold lines at 10-yard marks. `_field_to_canvas` maps `(lat_yd, lon_yd)`
to canvas pixels honoring the orientation toggle.

## Changes to existing files

### `server/utils/tracking_observer.py`

Add two new lifecycle hooks (both default no-op):

```python
class TrackingObserver:
    def on_session_start(self, owner: "Tracking") -> None: ...
    def on_session_end(self, owner: "Tracking") -> None: ...
```

These are needed so observers that own external resources (a separate
`VideoWriter`, a model, an outbound API connection) can init/release at the
session level, not per-play.

### `server/utils/tracking.py`

Three localized changes:

1. **Split `_DefaultDrawer`** so the heatmap responsibility is its own
   observer:

   ```python
   class _TrackedFrameDrawer(TrackingObserver):
       # Handles annotated_frame only (the existing box+ID overlay).
       ...

   class _OriginalHeatmapDrawer(TrackingObserver):
       # The existing behavior: circles + intra-team lines on a black
       # canvas in original-frame coordinates. Owns its own VideoWriter.
       def on_session_start(self, owner):
           self._writer = cv2.VideoWriter(
               str(RESULTS_DIR / RESULTS.heatmap_video_raw),
               cv2.VideoWriter_fourcc(*"mp4v"),
               owner.fps, (owner.w, owner.h),
           )
       def on_tracking_results(self, frame): ...     # existing draw logic
       def on_frame_drawn(self, frame):
           if self._writer is not None and frame.heatmap_frame is not None:
               self._writer.write(frame.heatmap_frame)
       def on_session_end(self, owner):
           if self._writer is not None: self._writer.release()
   ```

   The existing `_VideoSink` is unchanged — it continues to own the tracked
   video writer only.

2. **Add `replace_heatmap_drawer(observer)`** to `Tracking`:

   ```python
   def replace_heatmap_drawer(self, observer: TrackingObserver) -> None:
       # Swap whichever heatmap drawer is currently registered for this one.
       idx = self._observers.index(self._heatmap_drawer)
       self._observers[idx] = observer
       self._heatmap_drawer = observer
   ```

3. **Wire the new lifecycle hooks** at the start/end of `track_in_each_play`:

   ```python
   def track_in_each_play(self):
       self.initialize_video_handling()
       for o in self._observers: o.on_session_start(self)
       try:
           ... existing play loop ...
       finally:
           for o in self._observers: o.on_session_end(self)
       ... existing finalization ...
   ```

   And **remove** the heatmap writer init/release from
   `initialize_video_handling` and the play loop — that responsibility now
   lives on `_OriginalHeatmapDrawer` (or whichever drawer replaces it).

### `server/config.py`

Add three frozen dataclasses + singletons (same pattern as `TRACKER` / `HUDDLE`):

```python
@dataclass(frozen=True)
class BallCarrierConfig:
    conf_threshold:      float = 0.4
    iou_match_threshold: float = 0.3

BALLCARRIER: BallCarrierConfig = BallCarrierConfig()
# (PerspectiveConfig and BirdsEyeConfig live next to the observer that uses
# them in birds_eye_heatmap.py — no global singleton, the caller passes one
# in to BirdsEyeHeatmapObserver.)
```

## How the user switches between options

Both toggles are single-field config edits. Example for switching modes
side by side:

```python
from utils.tracking_refiner import RefinedTracking
from utils.observers.ball_carrier import BallCarrierObserver
from utils.observers.birds_eye_heatmap import (
    BirdsEyeHeatmapObserver, BirdsEyeConfig, PerspectiveConfig,
)

tracker = RefinedTracking(csv_name, frame_handler)

# --- Calibration cadence toggle ---
# Edit this one line:
persp_cfg = PerspectiveConfig(mode="cached", cache_every_n_frames=5)
# OR
persp_cfg = PerspectiveConfig(mode="every_frame")

# --- Orientation toggle ---
# Edit this one line:
birds_eye_cfg = BirdsEyeConfig(orientation="portrait")
# OR
birds_eye_cfg = BirdsEyeConfig(orientation="landscape")

tracker.replace_heatmap_drawer(
    BirdsEyeHeatmapObserver(tracker, birds_eye_cfg, persp_cfg)
)
tracker.register_observer(BallCarrierObserver(tracker))
tracker.track_in_each_play()
```

The user runs both modes by editing the corresponding line and re-running.

## Critical files

- **Created**: `server/utils/perspective.py`, `server/utils/observers/{__init__,ball_carrier,birds_eye_heatmap}.py`.
- **Modified**: `server/utils/tracking.py` (split drawer, add `replace_heatmap_drawer`, wire session hooks), `server/utils/tracking_observer.py` (add session hooks), `server/config.py` (add `BALLCARRIER`).
- **Read-only references**: `server/perspective_shift/pixel_yard_calib.ipynb` and `server/perspective_shift/pixel2yard_calibration.py` (algorithm source — re-implemented in `perspective.py` without `plt.show()` side effects).
- **Not touched**: every other server file. `tracking_refiner/` is unaffected; `RefinedTracking` continues to work, and the user can compose all three features (refiner + bird's-eye + ball carrier) by registering each observer.

## Cost optimization recommendation

The expensive parts of the perspective pipeline are:

| step                                | cost  | how to amortize                              |
|-------------------------------------|-------|----------------------------------------------|
| HSV mask + Hough line detection     | ~10ms | cache: recompute every K frames              |
| `curve_fit` Möbius                  | ~5ms  | same cache                                   |
| `cv2.findHomography(RANSAC)`        | ~3ms  | same cache                                   |
| `cv2.warpPerspective` full frame    | ~40ms | **skip entirely** — we only need `H`         |

The single largest saving is **not** warping the frame. We never need the
bird's-eye *image*; we need the bird's-eye *coordinates*. The synthetic green
canvas is drawn directly, and players are placed via `H @ [x, y, 1]` — one
3×3 matrix-vector product per player per frame, microseconds.

Second saving is the `CalibrationCache` (every-K-frames mode). At K=5 with
60fps source, calibration runs at ~12 Hz, fast enough to track even brisk
camera pans.

## Verification

1. **Unit-level (no video)**:
   - Hand-build a known `Calibration` (use the homography from the notebook's
     test image). Feed `pixel_to_field` known reference points; assert the
     yard coordinates match the notebook's spot-check table (lat≈26.67 at
     midfield) to ±0.5 yd.
   - Stub `Calibration` + a few synthetic player boxes through
     `BirdsEyeHeatmapObserver.on_tracking_results`; assert pixels land in
     the expected canvas region (left half for lat<32, etc.).
   - Smoke-test `BallCarrierObserver` with mock YOLO outputs returning (a) a
     valid box that IoU-matches one tracker tid, (b) no detection — assert
     `play.notes["ballcarrier_tid"]` updates correctly and persists across
     gap frames.

2. **Pipeline-level**: run on `server/data/videos/NFL_Blitz/AEAm6KANo1Q.mp4`
   with the existing `#14.csv`. Inspect `heatmap_video.mp4`:
   - Green background with the 5-yd grid visible.
   - Roughly 11 attackers (red) + 11 defenders (green outlined in white) +
     a single pink dot tracking the ball carrier through the play.
   - Dots track the same yard position even as the camera pans.

3. **Cost A/B**:
   - Run with `PerspectiveConfig(mode="every_frame")` then with
     `PerspectiveConfig(mode="cached", cache_every_n_frames=5)`. Measure
     wall-clock difference; user will pick visually.

4. **Orientation A/B**: same A/B with `orientation="portrait"` then
   `"landscape"`. User picks visually.

## Out of scope

- Real-time field-width measurement (the post-warp horizontal+vertical line
  detection from the notebook's last cells). We use the user's hardcoded 65
  yd ± 10 yd buffer for now. Adding measured-width later is one method on
  `CalibrationCache` and an update to canvas dims; no architectural change.
- OCR-anchored absolute longitudinal yard numbering. The notebook + existing
  `final_pixel2yard_calibration.py` already explored this and the user
  reports it performs poorly — skipped.
- Persisting calibration across plays. Camera angle can shift dramatically
  between plays, so the cache resets at each `on_play_start`.