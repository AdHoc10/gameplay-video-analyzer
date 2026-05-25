"""
Central project config.

Backwards-compatible: the original top-level constants (`GAME`, `VIDEOS_DIR`,
`START_TIMES_ANNOT_FOLDER`, `MOVE_ANNOT_FOLDER`) are still exported, so all
existing `from config import *` callers keep working.

New additions group by purpose so they can be imported individually:
    from config import TrackerConfig, HuddleConfig, ResultsConfig
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Existing globals — preserved for backwards compatibility.
# ---------------------------------------------------------------------------

GAME: str = "NFL_Blitz"
VIDEOS_DIR: str = f"./data/videos/{GAME}"
START_TIMES_ANNOT_FOLDER: str = f"./data/Annotations/start_times/{GAME}"
MOVE_ANNOT_FOLDER: str = f"./data/Annotations/actions/{GAME}"


# ---------------------------------------------------------------------------
# New: project root + structured path table
# ---------------------------------------------------------------------------

SERVER_ROOT: Path = Path(__file__).resolve().parent
"""Absolute path of the `server/` directory regardless of cwd."""

# Allow env overrides for production deployments where state lives elsewhere.
def _path(env_key: str, default: Path) -> Path:
    val = os.environ.get(env_key)
    return Path(val) if val else default


MODELS_DIR:  Path = _path("GVA_MODELS_DIR",  SERVER_ROOT / "models")
RESULTS_DIR: Path = _path("GVA_RESULTS_DIR", SERVER_ROOT / "results")
DATA_DIR:    Path = _path("GVA_DATA_DIR",    SERVER_ROOT / "data")
OUTPUTS_DIR: Path = _path("GVA_OUTPUTS_DIR", SERVER_ROOT / "outputs")


# ---------------------------------------------------------------------------
# Tracker config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackerConfig:
    """Knobs that used to live as bare numbers inside Tracking.__init__."""

    yolo_weights:        str = str(MODELS_DIR / "yolo11m.pt")
    bytetrack_yaml:      str = str(MODELS_DIR / "my_bytetrack.yaml")
    img_size:            int = 1280
    conf_threshold:      float = 0.15
    iou_threshold:       float = 0.5
    classes:             tuple = (0,)         # 0 = person in COCO


@dataclass(frozen=True)
class HuddleConfig:
    """Constants for huddle detection + per-play tag assignment."""

    huddle_weights:      str   = str(MODELS_DIR / "best_huddle_detection.pt")
    ballcarrier_weights: str   = str(MODELS_DIR / "best_ballCarrier_detection.pt")
    min_players:         int   = 9          # find_proper_huddle_frame
    tag_iou_threshold:   float = 0.3        # IoU to lock A/D tag to a track
    pre_snap_window_s:   float = 0.5        # seek-back before snap (seconds)


@dataclass(frozen=True)
class ResultsConfig:
    """Filenames the tracker emits into `RESULTS_DIR`."""

    tracked_video_raw:   str = "tracked_video_raw.mp4"
    tracked_video:       str = "tracked_video.mp4"
    heatmap_video_raw:   str = "heatmap_video_raw.mp4"
    heatmap_video:       str = "heatmap_video.mp4"
    timeline_map_json:   str = "tracked_timeline_map.json"


@dataclass(frozen=True)
class BallCarrierConfig:
    """Knobs for `BallCarrierObserver`."""

    conf_threshold:      float = 0.4   # min YOLO confidence to accept a bbox
    iou_match_threshold: float = 0.3   # min IoU to bind the bc bbox to a tracker tid


# Singletons for the common case — callers that want overrides can construct
# their own dataclass instance.
TRACKER:     TrackerConfig     = TrackerConfig()
HUDDLE:      HuddleConfig      = HuddleConfig()
RESULTS:     ResultsConfig     = ResultsConfig()
BALLCARRIER: BallCarrierConfig = BallCarrierConfig()
