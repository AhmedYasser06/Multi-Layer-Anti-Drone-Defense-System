"""
Layer 4 — NMSE-based Priority Targeting.

Definition used here (matches the report's Section 8.5 factor list):
For each tracked target we build a 4-feature vector, normalize every
feature to [0, 1] where 1 = "maximally threatening", then score how close
that vector is to the ideal maximum-threat vector (1,1,1,1) using a
*weighted normalized squared error*:

    score = 1 - sum_i( w_i * (1 - f_i)^2 )      , sum_i(w_i) = 1

This is exactly a normalized MSE between the target's feature vector and
the "worst case" reference vector, inverted so score=1 means highest
threat and score=0 means lowest. It naturally punishes targets that are
weak on any single high-weight factor (squared term) rather than just
averaging, which is what you want for triage: a target that's very close
but barely a confident detection shouldn't outrank a fast, close,
high-confidence one just because of a simple average.

Features:
  f_distance : 1 - (dist_to_center / frame_diagonal)      -> closer = higher
  f_speed    : closing speed toward protected-zone center, normalized
  f_conf     : fused YOLO confidence directly (already 0-1)
  f_size     : bbox area / frame area, saturating at BBOX_AREA_SATURATION_FRACTION
"""

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from . import config
from .tracker import Track


@dataclass
class PriorityEntry:
    score: float
    track_id: int
    cls_name: str


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_features(track: Track, frame_w: int, frame_h: int) -> Dict[str, float]:
    cx0, cy0 = frame_w / 2.0, frame_h / 2.0
    frame_diag = math.hypot(frame_w, frame_h)

    dist = math.hypot(track.cx - cx0, track.cy - cy0)
    f_distance = _clamp01(1.0 - dist / frame_diag)

    # Closing speed: positive when moving toward the center.
    to_center_x = cx0 - track.cx
    to_center_y = cy0 - track.cy
    to_center_mag = math.hypot(to_center_x, to_center_y) or 1e-6
    unit_x, unit_y = to_center_x / to_center_mag, to_center_y / to_center_mag
    closing_speed = track.velocity_x * unit_x + track.velocity_y * unit_y
    # Normalize against an assumed max meaningful speed of ~frame_w px/sec
    f_speed = _clamp01((closing_speed / frame_w) + 0.5)  # 0.5 baseline = "stationary"

    f_conf = _clamp01(track.conf)

    area = max(0.0, track.x2 - track.x1) * max(0.0, track.y2 - track.y1)
    frame_area = frame_w * frame_h
    f_size = _clamp01(
        area / (frame_area * config.BBOX_AREA_SATURATION_FRACTION)
    )

    return {
        "distance_to_center": f_distance,
        "closing_speed": f_speed,
        "confidence": f_conf,
        "bbox_size": f_size,
    }


def nmse_score(features: Dict[str, float]) -> float:
    err = 0.0
    for key, w in config.PRIORITY_WEIGHTS.items():
        f = features.get(key, 0.0)
        err += w * (1.0 - f) ** 2
    return _clamp01(1.0 - err)


class PriorityQueue:
    """Max-priority queue over live tracks, rebuilt each frame from current
    track state. Non-threat classes (Bird/AirPlane/Helicopter) are still
    tracked/displayed but never selected as the engage target."""

    def __init__(self):
        self._heap: List[Tuple[float, int, str]] = []

    def rebuild(self, tracks: Dict[int, Track], frame_w: int, frame_h: int):
        self._heap = []
        for tid, tr in tracks.items():
            if tr.cls_name not in config.THREAT_CLASSES:
                continue
            feats = compute_features(tr, frame_w, frame_h)
            score = nmse_score(feats)
            # heapq is a min-heap -> negate score for max-heap behavior
            heapq.heappush(self._heap, (-score, tid, tr.cls_name))

    def top_target(self) -> PriorityEntry | None:
        if not self._heap:
            return None
        neg_score, tid, cls_name = self._heap[0]
        return PriorityEntry(score=-neg_score, track_id=tid, cls_name=cls_name)

    def ranked(self) -> List[PriorityEntry]:
        return [
            PriorityEntry(score=-s, track_id=tid, cls_name=c)
            for s, tid, c in sorted(self._heap)
        ]
