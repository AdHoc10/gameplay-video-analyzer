"""
Per-play appearance gallery.

Two stores:

  * `active`  — currently-tracked IDs and their EMA embedding (fast cosine
                lookup for swap detection).
  * `lost`    — IDs that have stopped appearing recently. When a brand-new
                track ID shows up with no side tag, we cosine-match its
                embedding against `lost`; a hit recovers the original A/D tag.

State is intentionally per-play. The refiner constructs a fresh `TrackGallery`
at the start of each play and discards it at the end, so we never carry
appearance state across play boundaries (mirrors the existing
`predictor.trackers[0].reset()` semantics in tracking.py).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _LostEntry:
    emb: np.ndarray         # L2-normalized
    frame_lost: int
    side: Optional[str]     # "A" / "D" / None


@dataclass
class TrackGallery:
    """
    State container for one play's appearance gallery.

    Args:
        ema_alpha:        Weight given to the new sample when updating an
                          existing ID's embedding. 1.0 = no smoothing,
                          0.0 = ignore the new sample. 0.3 keeps history.
        reid_threshold:   Minimum cosine similarity to call a brand-new track
                          a re-emergence of a lost one.
        swap_threshold:   Minimum cosine similarity used in the bidirectional
                          swap check.
        max_lost_age:     Frames after which a lost entry is dropped.
        history_len:      How many raw embeddings to remember per active ID
                          (debugging only; not used in matching).
    """

    ema_alpha:       float = 0.3
    reid_threshold:  float = 0.78
    swap_threshold:  float = 0.85
    max_lost_age:    int   = 300
    history_len:     int   = 8

    # Internal state
    active:           Dict[int, np.ndarray]            = field(default_factory=dict)
    lost:             Dict[int, _LostEntry]            = field(default_factory=dict)
    last_seen_frame:  Dict[int, int]                   = field(default_factory=dict)
    history:          Dict[int, Deque[np.ndarray]]     = field(default_factory=dict)

    # ---- core ops ----------------------------------------------------------

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v if n < 1e-8 else (v / n).astype(np.float32)

    def _cos_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        # Both are unit-norm by construction; dot is cosine.
        return float(np.dot(a, b))

    def update_active(
        self,
        tid: int,
        emb: np.ndarray,
        frame_idx: int,
    ) -> None:
        """EMA-update (or insert) the active embedding for `tid`."""
        emb = self._l2(emb)
        if tid in self.active:
            prev = self.active[tid]
            merged = self.ema_alpha * emb + (1.0 - self.ema_alpha) * prev
            self.active[tid] = self._l2(merged)
        else:
            self.active[tid] = emb

        self.last_seen_frame[tid] = frame_idx
        h = self.history.setdefault(tid, deque(maxlen=self.history_len))
        h.append(emb)

    def mark_lost(
        self,
        tid: int,
        frame_idx: int,
        id_to_side: Dict[int, str],
    ) -> None:
        """Move an active ID into the lost gallery."""
        if tid not in self.active:
            return
        emb = self.active.pop(tid)
        self.lost[tid] = _LostEntry(
            emb=emb,
            frame_lost=frame_idx,
            side=id_to_side.get(tid),
        )
        # last_seen_frame & history kept for diagnostics; they don't affect
        # matching (which uses self.lost only).

    def match_lost(
        self,
        emb: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Optional[Tuple[int, float]]:
        """
        Find the best lost ID whose embedding matches `emb`.

        Returns `(tid_old, similarity)` if found, else None.
        """
        if not self.lost:
            return None
        thr = self.reid_threshold if threshold is None else threshold
        emb = self._l2(emb)

        best_tid: Optional[int] = None
        best_sim = -1.0
        for tid, entry in self.lost.items():
            sim = self._cos_sim(emb, entry.emb)
            if sim > best_sim:
                best_sim = sim
                best_tid = tid

        if best_tid is None or best_sim < thr:
            return None
        return best_tid, best_sim

    def pop_lost(self, tid: int) -> Optional[_LostEntry]:
        """Remove and return a lost entry (used after a successful re-id)."""
        return self.lost.pop(tid, None)

    # ---- swap detection ----------------------------------------------------

    def detect_swaps(
        self,
        tid_emb_pairs: Iterable[Tuple[int, np.ndarray]],
    ) -> List[Tuple[int, int]]:
        """
        Identify pairs (a, b) where the current embedding of `a` matches the
        active gallery entry of `b` *and* vice-versa, both above
        `swap_threshold`.

        Only considers IDs that already exist in `self.active` — we are
        catching ID-swap-during-overlap, not new-ID re-id.
        """
        # Build {tid: current_emb} for tids that already have a gallery entry.
        current: Dict[int, np.ndarray] = {}
        for tid, emb in tid_emb_pairs:
            if tid in self.active:
                current[tid] = self._l2(emb)
        if len(current) < 2:
            return []

        # For each current emb, find which gallery entry it matches best.
        best_match: Dict[int, Tuple[int, float]] = {}
        for tid, cur in current.items():
            best_other = None
            best_sim = -1.0
            for other_tid, gal_emb in self.active.items():
                sim = self._cos_sim(cur, gal_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_other = other_tid
            if best_other is not None:
                best_match[tid] = (best_other, best_sim)

        swaps: List[Tuple[int, int]] = []
        seen: set = set()
        for tid_a, (tid_b, sim_ab) in best_match.items():
            if tid_a == tid_b or tid_a in seen or tid_b in seen:
                continue
            other = best_match.get(tid_b)
            if other is None:
                continue
            tid_back, sim_ba = other
            if tid_back != tid_a:
                continue
            if sim_ab < self.swap_threshold or sim_ba < self.swap_threshold:
                continue
            # Mutual best match between two distinct active IDs above threshold
            # — that's an ID swap.
            swaps.append((tid_a, tid_b))
            seen.add(tid_a)
            seen.add(tid_b)
        return swaps

    # ---- housekeeping ------------------------------------------------------

    def prune(self, frame_idx: int) -> None:
        """Drop lost entries older than `max_lost_age` frames."""
        if not self.lost:
            return
        cutoff = frame_idx - self.max_lost_age
        stale = [tid for tid, e in self.lost.items() if e.frame_lost < cutoff]
        for tid in stale:
            self.lost.pop(tid, None)

    def reset(self) -> None:
        self.active.clear()
        self.lost.clear()
        self.last_seen_frame.clear()
        self.history.clear()
