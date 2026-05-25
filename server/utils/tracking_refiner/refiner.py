"""
Per-play tracking refiner.

Glues an `Embedder` and a `TrackGallery` together. The refiner has no I/O,
no drawing, no video writes — its only job is to mutate the tracker's
`id_to_side` map so downstream code (drawing, defender counts) sees the
correct A/D tag for every track ID, even after a track reset, occlusion,
or ID swap.

Lifecycle (one play):
    refiner.reset()                       # at top of the play
    for frame_idx in play_frames:
        ...
        refiner.refine(
            frame_rgb  = ...,             # HxWx3 uint8 RGB
            boxes_xyxy = ndarray (N, 4),  # ints / floats
            tids       = ndarray (N,),    # track IDs
            id_to_side = id_to_side,      # mutated in place
            frame_idx  = frame_idx,
        )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedders import Embedder
from .gallery import TrackGallery

logger = logging.getLogger(__name__)


@dataclass
class RefinerConfig:
    """Tunable knobs for the refiner / gallery."""

    # ---- gallery thresholds ----
    reid_threshold:  float = 0.78
    swap_threshold:  float = 0.85
    ema_alpha:       float = 0.3
    max_lost_age:    int   = 300        # frames

    # ---- per-frame behaviour ----
    lost_grace:      int   = 3          # absent frames before mark_lost
    min_crop_area:   int   = 32 * 32    # px²; smaller crops are skipped
    embed_every_n_frames: int = 1       # increase to amortize cost

    # ---- diagnostics ----
    log_remaps:      bool  = False


@dataclass
class RefineOutput:
    """What the refiner did this frame."""

    remapped_tids: Dict[int, int]     = field(default_factory=dict)
    """current_tid -> previously-known tid that owned the same identity."""

    swaps: List[Tuple[int, int]]      = field(default_factory=list)
    """ID-swap pairs corrected this frame."""

    embedded: bool                    = False
    """Did we actually run the embedder this frame?"""


class TrackingRefiner:
    """Stateful, single-play refiner. Build once per play, call `refine` per frame."""

    def __init__(
        self,
        embedder: Embedder,
        config: Optional[RefinerConfig] = None,
    ):
        self.embedder = embedder
        self.config = config or RefinerConfig()
        self.gallery = self._fresh_gallery()
        self._frames_seen = 0

    # ---- lifecycle ---------------------------------------------------------

    def _fresh_gallery(self) -> TrackGallery:
        c = self.config
        return TrackGallery(
            ema_alpha=c.ema_alpha,
            reid_threshold=c.reid_threshold,
            swap_threshold=c.swap_threshold,
            max_lost_age=c.max_lost_age,
        )

    def reset(self) -> None:
        """Wipe gallery + counters. Call at the start of each play."""
        self.gallery = self._fresh_gallery()
        self._frames_seen = 0

    # ---- per-frame --------------------------------------------------------

    def refine(
        self,
        frame_rgb:  np.ndarray,
        boxes_xyxy: np.ndarray,
        tids:       np.ndarray,
        id_to_side: Dict[int, str],
        frame_idx:  int,
    ) -> RefineOutput:
        """
        Reconcile tracker output against the appearance gallery.

        Mutates `id_to_side` in place:
          * Newly-appeared IDs that match a recently-lost track inherit
            the lost track's side tag.
          * IDs identified as swapped exchange their side tags.
        """
        self._frames_seen += 1
        out = RefineOutput()

        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            # No detections this frame — only do housekeeping.
            self._handle_absentees(set(), frame_idx, id_to_side)
            self.gallery.prune(frame_idx)
            return out

        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
        tids = np.asarray(tids, dtype=int)

        # Skip frames per `embed_every_n_frames` (still do housekeeping).
        n = max(1, self.config.embed_every_n_frames)
        if (self._frames_seen - 1) % n != 0:
            self._handle_absentees(set(tids.tolist()), frame_idx, id_to_side)
            self.gallery.prune(frame_idx)
            return out

        # Filter out degenerate / tiny crops up front so we don't waste the
        # embedder on them, but keep index alignment with `tids`.
        keep_mask = self._area_mask(boxes_xyxy)
        if not keep_mask.any():
            self._handle_absentees(set(tids.tolist()), frame_idx, id_to_side)
            self.gallery.prune(frame_idx)
            return out

        # Single batched GPU call.
        embeddings_full = np.zeros(
            (len(boxes_xyxy), self.embedder.dim), dtype=np.float32
        )
        embeddings_full[keep_mask] = self.embedder.embed_batch(
            frame_rgb, boxes_xyxy[keep_mask]
        )
        out.embedded = True

        # 1) Re-attach side tags for brand-new IDs that match a lost track.
        for i, tid in enumerate(tids):
            tid = int(tid)
            if not keep_mask[i]:
                continue
            if id_to_side.get(tid) is not None:
                continue
            match = self.gallery.match_lost(embeddings_full[i])
            if match is None:
                continue
            tid_old, sim = match
            entry = self.gallery.pop_lost(tid_old)
            if entry is None or entry.side is None:
                continue
            id_to_side[tid] = entry.side
            out.remapped_tids[tid] = tid_old
            if self.config.log_remaps:
                logger.info(
                    "[refiner] frame=%d re-id: tid %d -> side %s (was tid %d, sim=%.3f)",
                    frame_idx, tid, entry.side, tid_old, sim,
                )

        # 2) Detect mutual-best ID swaps among currently active tracks.
        pairs = [
            (int(tids[i]), embeddings_full[i])
            for i in range(len(tids))
            if keep_mask[i] and int(tids[i]) in self.gallery.active
        ]
        if len(pairs) >= 2:
            swaps = self.gallery.detect_swaps(pairs)
            for a, b in swaps:
                side_a = id_to_side.get(a)
                side_b = id_to_side.get(b)
                if side_a == side_b:
                    # Same tag (or both missing) — nothing to fix.
                    continue
                # Only swap when both sides are concretely known; otherwise we
                # could overwrite a good tag with None.
                if side_a is None or side_b is None:
                    continue
                id_to_side[a], id_to_side[b] = side_b, side_a
                out.swaps.append((a, b))
                if self.config.log_remaps:
                    logger.info(
                        "[refiner] frame=%d swap-fix: %d<->%d (sides %s<->%s)",
                        frame_idx, a, b, side_a, side_b,
                    )

        # 3) Update the active gallery with the current observations.
        for i, tid in enumerate(tids):
            if not keep_mask[i]:
                continue
            self.gallery.update_active(int(tid), embeddings_full[i], frame_idx)

        # 4) Move long-missing IDs into the lost gallery.
        self._handle_absentees(set(int(t) for t in tids), frame_idx, id_to_side)
        self.gallery.prune(frame_idx)
        return out

    # ---- helpers ----------------------------------------------------------

    def _area_mask(self, boxes: np.ndarray) -> np.ndarray:
        w = np.maximum(0, boxes[:, 2] - boxes[:, 0])
        h = np.maximum(0, boxes[:, 3] - boxes[:, 1])
        return (w * h) >= self.config.min_crop_area

    def _handle_absentees(
        self,
        seen_tids: set,
        frame_idx: int,
        id_to_side: Dict[int, str],
    ) -> None:
        """Promote active IDs absent for > lost_grace frames into the lost set."""
        cutoff = frame_idx - self.config.lost_grace
        # Snapshot keys to mutate dict during iteration.
        for tid in list(self.gallery.active.keys()):
            if tid in seen_tids:
                continue
            last = self.gallery.last_seen_frame.get(tid, frame_idx)
            if last < cutoff:
                self.gallery.mark_lost(tid, frame_idx, id_to_side)
