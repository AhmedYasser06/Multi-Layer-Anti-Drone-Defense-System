"""
Multi-object tracker that assigns persistent IDs to fused detections across
frames. Ships with a lightweight centroid+IoU tracker (zero extra
dependencies, good enough for a handful of simultaneous targets at
25-30 FPS). If you want to swap in real Deep SORT (appearance embeddings,
better under occlusion), install `deep-sort-realtime` and see the
DeepSortTracker class at the bottom -- same interface, drop-in replacement.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List

from . import config
from .detection_source import NormalizedDetection


@dataclass
class Track:
    track_id: int
    cx: float
    cy: float
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_name: str
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    misses: int = 0
    last_update_t: float = field(default_factory=time.time)
    lock_streak: int = 0  # consecutive frames within LOCK_PIXEL_RADIUS of frame center


class CentroidIoUTracker:
    def __init__(self):
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def _iou(self, a: Track, d: NormalizedDetection) -> float:
        ix1, iy1 = max(a.x1, d.x1), max(a.y1, d.y1)
        ix2, iy2 = min(a.x2, d.x2), min(a.y2, d.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
        area_d = max(0.0, d.x2 - d.x1) * max(0.0, d.y2 - d.y1)
        union = area_a + area_d - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: List[NormalizedDetection]) -> Dict[int, Track]:
        now = time.time()
        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        # Greedy matching: for each existing track, pick the best remaining
        # detection above an IoU floor. Fine for small numbers of targets.
        for tid, tr in list(self.tracks.items()):
            best_j, best_iou = -1, 0.15  # IoU floor to accept a match
            for j in unmatched_dets:
                iou = self._iou(tr, detections[j])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j != -1:
                d = detections[best_j]
                dt = max(1e-3, now - tr.last_update_t)
                tr.velocity_x = (d.cx - tr.cx) / dt
                tr.velocity_y = (d.cy - tr.cy) / dt
                tr.cx, tr.cy = d.cx, d.cy
                tr.x1, tr.y1, tr.x2, tr.y2 = d.x1, d.y1, d.x2, d.y2
                tr.conf = d.fused_conf
                tr.cls_name = d.cls_name
                tr.misses = 0
                tr.last_update_t = now
                unmatched_dets.remove(best_j)
                matched_track_ids.add(tid)
            else:
                tr.misses += 1

        # New tracks for leftover detections
        for j in unmatched_dets:
            d = detections[j]
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(
                track_id=tid, cx=d.cx, cy=d.cy,
                x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                conf=d.fused_conf, cls_name=d.cls_name,
            )

        # Drop stale tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].misses > config.TRACK_LOST_FRAMES:
                del self.tracks[tid]

        return self.tracks


# ---------------------------------------------------------------------------
# Optional: real Deep SORT backend. Requires: pip install deep-sort-realtime
# Same public interface (`update(fused_detections) -> Dict[int, Track]`) so
# main.py doesn't need to change if you switch trackers.
# ---------------------------------------------------------------------------
class DeepSortTracker:
    def __init__(self):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self._ds = DeepSort(max_age=config.TRACK_LOST_FRAMES)
        self._last_seen_t = {}

    def update(self, detections: List[NormalizedDetection], frame=None) -> Dict[int, Track]:
        # deep_sort_realtime expects ([x1,y1,w,h], conf, class) tuples and
        # the raw frame (it extracts appearance embeddings via a CNN crop).
        raw = [
            ([d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1], d.fused_conf, d.cls_name)
            for d in detections
        ]
        ds_tracks = self._ds.update_tracks(raw, frame=frame)
        now = time.time()
        out: Dict[int, Track] = {}
        for t in ds_tracks:
            if not t.is_confirmed():
                continue
            tid = int(t.track_id)
            l, top, r, b = t.to_ltrb()
            cx, cy = (l + r) / 2, (top + b) / 2
            prev_t = self._last_seen_t.get(tid, now)
            dt = max(1e-3, now - prev_t)
            prev = out.get(tid)
            vx = vy = 0.0
            out[tid] = Track(
                track_id=tid, cx=cx, cy=cy, x1=l, y1=top, x2=r, y2=b,
                conf=t.get_det_conf() or 0.0,
                cls_name=t.get_det_class() or "unknown",
                velocity_x=vx, velocity_y=vy,
            )
            self._last_seen_t[tid] = now
        return out
