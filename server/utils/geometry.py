"""
Small geometry helpers shared across the tracker, refiner, and any future
feature that consumes bounding boxes.

Keep this file free of heavy deps (cv2, torch, ultralytics). It should be
safe to import from anywhere.
"""

from __future__ import annotations

from typing import Sequence


def iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """
    Axis-aligned IoU for boxes in xyxy format.

    Args:
        b1, b2: 4-element sequences [x1, y1, x2, y2].

    Returns:
        IoU in [0, 1]. Returns 0 for non-overlapping or degenerate boxes.
    """
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return float(inter / (union + 1e-6))
