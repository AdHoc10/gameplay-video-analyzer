"""
Per-play YOLO + ByteTrack tracker with pluggable observers.

The previous monolithic `track_in_each_play` is now decomposed into:

  * a slim main loop that emits lifecycle events
  * built-in observers that ship with `Tracking`:
        - `_HuddleTagger`         → A/D tags on snap frame
        - `_TrackedFrameDrawer`   → boxes + IDs on `annotated_frame`
        - `_OriginalHeatmapDrawer`→ original-frame heatmap (own VideoWriter)
        - `_TrackedVideoSink`     → cv2.VideoWriter for the tracked video
        - `_EventCollector`       → defender counts + frames-since-snap
        - `_TimelineMapper`       → source<->tracked-frame segments
  * a `register_observer()` API so new features (refiner, ball-carrier,
    bird's-eye heatmap, …) plug in without subclassing the whole loop
  * a `replace_heatmap_drawer()` API to swap the heatmap renderer for a
    different one (e.g., the bird's-eye drawer)

Behaviour is bit-identical to the original when no extra observers are
registered and the heatmap drawer is not replaced.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from config import HUDDLE, RESULTS, RESULTS_DIR, TRACKER
from utils.frames_process import FrameHandler
from utils.geometry import iou as _iou
from utils.h264_conversion import convert_to_h264
from utils.tracking_observer import FrameContext, PlayContext, TrackingObserver


# ---------------------------------------------------------------------------
# Built-in observers
# ---------------------------------------------------------------------------

class _HuddleTagger(TrackingObserver):
    """Assign A/D tags on the snap frame by IoU-matching against the
    fine-tuned huddle detection model."""

    def __init__(self, owner: "Tracking"):
        self.owner = owner

    def on_tracking_results(self, frame: FrameContext) -> None:
        if not frame.is_snap_frame:
            return
        snap = self.owner._snap_results
        if snap is None or snap[0].boxes.id is None:
            return

        trk_boxes = snap[0].boxes.xyxy.cpu().numpy().astype(int)
        trk_ids   = snap[0].boxes.id.cpu().numpy().astype(int)
        hdl_res   = self.owner.huddle_detection_model(frame.frame_bgr, verbose=False)[0]

        hdl_boxes: List[Tuple[np.ndarray, str]] = []
        for b in hdl_res.boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            cls  = int(b.cls[0])           # 0 = attacker, 1 = defender
            hdl_boxes.append((xyxy, "A" if cls == 0 else "D"))

        for tb, tid in zip(trk_boxes, trk_ids):
            best_iou, best_side = 0.0, None
            for hb, side in hdl_boxes:
                v = _iou(tb, hb)
                if v > best_iou:
                    best_iou, best_side = v, side
            if best_iou > HUDDLE.tag_iou_threshold:
                frame.play.id_to_side[int(tid)] = best_side  # type: ignore[assignment]
                frame.play.id_to_initial_y[int(tid)] = float((tb[1] + tb[3]) / 2.0)


class _TrackedFrameDrawer(TrackingObserver):
    """Paint boxes + IDs onto `annotated_frame` (the tracked-video output).

    Heatmap rendering lives in a separate observer so it can be swapped
    (e.g. for the bird's-eye drawer).

    Ball-carrier priority: if `BallCarrierObserver` is registered it writes
    `frame.play.notes["ballcarrier_tid"]` before this observer runs (the
    `register_observer` API inserts external observers before this drawer).
    The ball carrier's box is rendered in pink regardless of their A/D side
    tag, and their label gets a "BC" prefix so the carrier is immediately
    recognisable in the tracked video.
    """

    # BGR pink — matches the heatmap drawer's `bc_color_bgr`.
    _BC_COLOR: Tuple[int, int, int] = (180, 105, 255)

    def on_tracking_results(self, frame: FrameContext) -> None:
        if frame.annotated_frame is None:
            frame.annotated_frame = frame.frame_bgr.copy()
        if frame.results is None or frame.results[0].boxes.id is None:
            return

        boxes      = frame.results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids        = frame.results[0].boxes.id.cpu().numpy().astype(int)
        id_to_side = frame.play.id_to_side
        bc_tid     = frame.play.notes.get("ballcarrier_tid")

        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            tid_int = int(tid)
            if bc_tid is not None and tid_int == bc_tid:
                # Ball carrier — pink box, "BC" prefix, regardless of side.
                side   = id_to_side.get(tid_int)
                suffix = f" ({('ATT' if side == 'A' else 'DEF') if side else 'UNK'})"
                colour = self._BC_COLOR
                label  = f"BC {tid_int}{suffix}"
            else:
                side = id_to_side.get(tid_int)
                if side is None:
                    colour, label = (255, 255, 255), f"UNK {tid_int}"
                else:
                    colour = (0, 255, 0) if side == "A" else (0, 0, 255)
                    label  = f"{'ATT' if side == 'A' else 'DEF'} {tid_int}"
            cv2.rectangle(frame.annotated_frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame.annotated_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


class _OriginalHeatmapDrawer(TrackingObserver):
    """Default heatmap renderer: per-team circles + intra-team lines on a
    black canvas in *original-frame* coordinates. Owns its own VideoWriter.

    This is what existed before the bird's-eye drawer was introduced —
    kept here so the existing pipeline behaviour is preserved unless the
    caller swaps it via `Tracking.replace_heatmap_drawer`.
    """

    def __init__(self):
        self._writer: Optional[cv2.VideoWriter] = None
        self._raw_path:  Optional[str] = None
        self._final_path: Optional[str] = None

    def on_session_start(self, owner: "Tracking") -> None:
        self._raw_path   = str(RESULTS_DIR / RESULTS.heatmap_video_raw)
        self._final_path = str(RESULTS_DIR / RESULTS.heatmap_video)
        self._writer = cv2.VideoWriter(
            self._raw_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            owner.fps,
            (owner.w, owner.h),
        )
        # Expose paths on the owner so the main loop's finalization step can
        # find them regardless of which heatmap drawer is active.
        owner.heatmap_video_raw  = self._raw_path
        owner.heatmap_video_path = self._final_path

    def on_tracking_results(self, frame: FrameContext) -> None:
        if frame.heatmap_frame is None:
            h, w = frame.frame_bgr.shape[:2]
            frame.heatmap_frame = np.zeros((h, w, 3), dtype=np.uint8)
        if frame.results is None or frame.results[0].boxes.id is None:
            return

        boxes = frame.results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids   = frame.results[0].boxes.id.cpu().numpy().astype(int)
        id_to_side = frame.play.id_to_side

        team_side_coordinates: Dict[str, List[Tuple[int, int]]] = {"A": [], "D": []}
        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            side = id_to_side.get(int(tid))
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if side is not None:
                colour = (0, 255, 0) if side == "A" else (0, 0, 255)
                cv2.circle(frame.heatmap_frame, (cx, cy), 7, colour, -1)
                team_side_coordinates[side].append((cx, cy))

        for side, pts in team_side_coordinates.items():
            if len(pts) < 2:
                continue
            line_colour = (0, 255, 0) if side == "A" else (0, 0, 255)
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    cv2.line(frame.heatmap_frame, pts[i], pts[j],
                             line_colour, 2, lineType=cv2.LINE_AA)

    def on_frame_drawn(self, frame: FrameContext) -> None:
        if self._writer is not None and frame.heatmap_frame is not None:
            self._writer.write(frame.heatmap_frame)

    def on_session_end(self, owner: "Tracking") -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class _TrackedVideoSink(TrackingObserver):
    """Owns the cv2.VideoWriter for the tracked (annotated) video."""

    def __init__(self, owner: "Tracking"):
        self.owner = owner

    def on_frame_drawn(self, frame: FrameContext) -> None:
        if frame.annotated_frame is not None:
            self.owner.out.write(frame.annotated_frame)


class _EventCollector(TrackingObserver):
    """Builds the `moves_2_defenderCount_dict` and
    `moves_2_timeSincePlayBegan` dicts that feed P(move|...)."""

    def __init__(self):
        self.moves_2_defenderCount_dict: Dict[int, tuple] = {}
        self.moves_2_timeSincePlayBegan: Dict[int, tuple] = {}

    def on_frame_drawn(self, frame: FrameContext) -> None:
        moves = frame.play.moves
        action_tags = [tag for tag, frm, _ in moves if frm == frame.frame_idx]
        if not action_tags:
            return
        if frame.results is None or frame.results[0].boxes.id is None:
            return
        down_nums = [down for _, frm, down in moves if frm == frame.frame_idx]
        defenders_count = list(frame.play.id_to_side.values()).count("D")
        for tag, down in zip(action_tags, down_nums):
            self.moves_2_defenderCount_dict[frame.frame_idx] = (tag, defenders_count)
            self.moves_2_timeSincePlayBegan[frame.frame_idx] = (
                tag, frame.frame_idx - frame.play.start_time_frame, down
            )


class _TimelineMapper(TrackingObserver):
    """Records {source_frame_range, tracked_frame_range} per play."""

    def __init__(self):
        self.segments: List[Dict[str, int]] = []
        self.output_frame_idx: int = 0

    def on_frame_drawn(self, frame: FrameContext) -> None:
        if frame.play.play_source_start is None:
            frame.play.play_source_start = frame.frame_idx
        frame.play.play_source_end = frame.frame_idx
        self.output_frame_idx += 1

    def on_play_end(self, play: PlayContext) -> None:
        if (play.play_source_start is not None
                and play.play_source_end is not None
                and self.output_frame_idx > play.play_output_start):
            self.segments.append({
                "source_start_frame":  int(play.play_source_start),
                "source_end_frame":    int(play.play_source_end),
                "tracked_start_frame": int(play.play_output_start),
                "tracked_end_frame":   int(self.output_frame_idx - 1),
            })


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Tracking:
    """Per-play YOLO + ByteTrack runner with pluggable observer hooks."""

    def __init__(self, csv_name: str, frame_handler: FrameHandler):
        # Models
        self.track_model: YOLO = YOLO(TRACKER.yolo_weights)
        self.huddle_detection_model: YOLO = YOLO(HUDDLE.huddle_weights)
        self.ballCarrier_detection_model: YOLO = YOLO(HUDDLE.ballcarrier_weights)
        self.tracker: str = TRACKER.bytetrack_yaml

        # Tracker config
        self.track_img_size: int   = TRACKER.img_size
        self.track_conf:     float = TRACKER.conf_threshold
        self.track_iou:      float = TRACKER.iou_threshold

        # Inputs
        self.frame_handler: FrameHandler = frame_handler

        # Observers — built-ins first, user-registered ones go after.
        self._observers: List[TrackingObserver] = []
        self._huddle_tagger:    _HuddleTagger          = _HuddleTagger(self)
        self._tracked_drawer:   _TrackedFrameDrawer    = _TrackedFrameDrawer()
        self._heatmap_drawer:   TrackingObserver       = _OriginalHeatmapDrawer()
        self._video_sink:       _TrackedVideoSink      = _TrackedVideoSink(self)
        self._event_collector:  _EventCollector        = _EventCollector()
        self._timeline_mapper:  _TimelineMapper        = _TimelineMapper()
        for o in (
            self._huddle_tagger,
            self._tracked_drawer,
            self._heatmap_drawer,
            self._video_sink,
            self._event_collector,
            self._timeline_mapper,
        ):
            self._observers.append(o)

        # Transient per-frame snap result for the huddle tagger.
        self._snap_results = None

        # Lazily set in `initialize_video_handling`; pre-declared for type hints.
        self.cap: Optional[cv2.VideoCapture] = None
        self.out: Optional[cv2.VideoWriter]  = None
        self.heatmap_video_raw:  Optional[str] = None
        self.heatmap_video_path: Optional[str] = None

    # ---- public API --------------------------------------------------------

    def register_observer(self, observer: TrackingObserver) -> None:
        """Register an additional observer.

        Order matters: observers fire in the order registered, so register
        anything that mutates `id_to_side` (refiner, swap-fix) BEFORE the
        drawers if you want the drawing to reflect the corrected tags.
        By default we insert before `_TrackedFrameDrawer` so that 'plug-in'
        observers automatically get this guarantee.
        """
        drawer_idx = self._observers.index(self._tracked_drawer)
        self._observers.insert(drawer_idx, observer)

    def replace_heatmap_drawer(self, observer: TrackingObserver) -> None:
        """Swap the heatmap drawer for a different one.

        The new observer takes over both the rendering and the VideoWriter
        ownership for the heatmap output. Use this to switch from the
        original-frame heatmap to e.g. the bird's-eye one:

            tracker.replace_heatmap_drawer(BirdsEyeHeatmapObserver(tracker))
        """
        idx = self._observers.index(self._heatmap_drawer)
        self._observers[idx] = observer
        self._heatmap_drawer = observer

    # ---- staticmethod kept for back-compat -------------------------------

    @staticmethod
    def iou(b1, b2) -> float:
        """Thin wrapper over `utils.geometry.iou` for any legacy caller."""
        return _iou(b1, b2)

    # ---- video plumbing ---------------------------------------------------

    def initialize_video_handling(self) -> None:
        self.cap = cv2.VideoCapture(self.frame_handler.video_path)
        self.w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.output_video_raw  = str(RESULTS_DIR / RESULTS.tracked_video_raw)
        self.output_video_path = str(RESULTS_DIR / RESULTS.tracked_video)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            self.output_video_raw, fourcc, self.fps, (self.w, self.h)
        )
        # heatmap_video_raw / heatmap_video_path are populated by whichever
        # heatmap drawer is active, in its on_session_start.

    # ---- main loop --------------------------------------------------------

    def track_in_each_play(
        self,
    ) -> Tuple[Dict[int, tuple], Dict[int, tuple], str, str, str]:
        self.initialize_video_handling()
        self._emit_session_start()

        pre_snap_frames = int(self.fps * HUDDLE.pre_snap_window_s)

        try:
            for play_index, (start_time_frame, moves) in enumerate(
                tqdm(self.frame_handler.huddle_to_moves_map.items())
            ):
                if not moves:
                    continue

                last_frame = moves[-1][1]
                play = PlayContext(
                    play_index=play_index,
                    start_time_frame=start_time_frame,
                    moves=list(moves),
                    play_output_start=self._timeline_mapper.output_frame_idx,
                )
                self._emit_play_start(play)

                frame_idx = start_time_frame - pre_snap_frames
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    # 1) Snap-frame detection (used by _HuddleTagger only).
                    if frame_idx == start_time_frame:
                        self._snap_results = self.track_model.track(
                            frame,
                            imgsz=self.track_img_size,
                            conf=self.track_conf,
                            iou=self.track_iou,
                            persist=False,
                            tracker=self.tracker,
                            classes=list(TRACKER.classes),
                            verbose=False,
                        )

                    # 2) Continue tracking with locked labels.
                    results = self.track_model.track(
                        frame,
                        imgsz=self.track_img_size,
                        conf=self.track_conf,
                        iou=self.track_iou,
                        persist=True,
                        tracker=self.tracker,
                        classes=list(TRACKER.classes),
                        verbose=False,
                    )

                    # 3) Build the frame context and fire observers.
                    fctx = FrameContext(
                        play=play,
                        frame_idx=frame_idx,
                        frame_bgr=frame,
                        results=results,
                        is_snap_frame=(frame_idx == start_time_frame),
                    )
                    self._emit_tracking_results(fctx)
                    self._emit_frame_drawn(fctx)

                    if frame_idx >= last_frame + self.fps:
                        break
                    frame_idx += 1

                self._emit_play_end(play)
                # Reset tracker between plays so IDs don't bleed across discontinuous segments.
                self.track_model.predictor.trackers[0].reset()
        finally:
            self._emit_session_end()
            if self.cap is not None:
                self.cap.release()
            if self.out is not None:
                self.out.release()

        # Timeline map JSON
        timeline_path = str(RESULTS_DIR / RESULTS.timeline_map_json)
        with open(timeline_path, "w", encoding="utf-8") as f:
            json.dump({
                "fps": int(self.fps),
                "segments": self._timeline_mapper.segments,
            }, f, indent=2)

        print("Converting tracked video to H.264 for browser playback...")
        convert_to_h264(self.output_video_raw,  self.output_video_path)
        if self.heatmap_video_raw and self.heatmap_video_path \
                and os.path.exists(self.heatmap_video_raw):
            print("Converting heatmap video to H.264 for browser playback...")
            convert_to_h264(self.heatmap_video_raw, self.heatmap_video_path)
            print(f"Saved → {self.heatmap_video_path}")
        print(f"Saved → {self.output_video_path}")

        return (
            self._event_collector.moves_2_defenderCount_dict,
            self._event_collector.moves_2_timeSincePlayBegan,
            self.output_video_path,
            self.heatmap_video_path or "",
            timeline_path,
        )

    # ---- observer dispatch -----------------------------------------------

    def _emit_session_start(self) -> None:
        for o in self._observers:
            o.on_session_start(self)

    def _emit_session_end(self) -> None:
        for o in self._observers:
            o.on_session_end(self)

    def _emit_play_start(self, play: PlayContext) -> None:
        for o in self._observers:
            o.on_play_start(play)

    def _emit_tracking_results(self, frame: FrameContext) -> None:
        for o in self._observers:
            o.on_tracking_results(frame)

    def _emit_frame_drawn(self, frame: FrameContext) -> None:
        for o in self._observers:
            o.on_frame_drawn(frame)

    def _emit_play_end(self, play: PlayContext) -> None:
        for o in self._observers:
            o.on_play_end(play)
