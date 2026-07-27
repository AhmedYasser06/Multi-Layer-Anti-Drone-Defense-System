"""
Layer 2/3 sensor fusion: RGB and thermal cameras now run SIMULTANEOUSLY,
every frame -- there is no more day/night switch that runs only one
camera+model pair at a time. Both feeds are always read, both models
always run, and their detections are merged here before anything reaches
the tracker.

Why fuse instead of just handing the tracker two independent detection
lists:
  - A drone that's washed out in RGB (dusk, backlight, glare, camouflage
    paint) can still carry a strong thermal signature from its motors,
    and vice versa (thermal false-positives like sun-heated rooftops
    don't show up as a bird/drone shape in RGB). Fusing raises confidence
    when both agree and still catches the target when only one sensor
    fires.
  - Without fusion, the same physical drone would spawn two overlapping
    tracks (one from the RGB camera's box, one from the thermal camera's
    box), which would confuse the tracker's ID assignment and double-
    count it in the priority queue.

How the two camera views are aligned:
  Thermal and RGB sensors are physically offset on the rig, so a pixel in
  the thermal frame is NOT at the same (x, y) in the RGB frame. We handle
  this the same way tools/calibrate_thermal_offset.py already documents:
    1. Preferred: a calibrated 2x3 similarity transform (rotation +
       uniform scale + translation), solved from several point
       correspondences, loaded from config.THERMAL_TRANSFORM_PATH.
    2. Fallback (no calibration file yet): a constant pixel offset from
       config.THERMAL_OFFSET_X / THERMAL_OFFSET_Y. Fine for a rough demo,
       but will drift near frame edges -- calibrate before real testing.
"""

import json
import os
from typing import List

import cv2
import numpy as np

from . import config
from .detection_source import NormalizedDetection


def _iou(a: NormalizedDetection, b: NormalizedDetection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_transform():
    """Returns a 2x3 np.float32 affine matrix, or None if no calibration
    file exists yet (caller should fall back to a constant offset)."""
    path = config.THERMAL_TRANSFORM_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return np.array(data["matrix_2x3"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as e:
        print(f"[FUSION] Could not read {path} ({e}); falling back to constant offset.")
        return None


class SensorFusion:
    def __init__(self):
        self._transform = _load_transform()
        if self._transform is None:
            print(
                "[FUSION] No calibrated thermal->RGB transform found at "
                f"{config.THERMAL_TRANSFORM_PATH}. Using constant offset "
                f"({config.THERMAL_OFFSET_X}, {config.THERMAL_OFFSET_Y}). "
                "Run `python -m tools.calibrate_thermal_offset` for real alignment."
            )
        else:
            print(f"[FUSION] Loaded calibrated thermal->RGB transform from {config.THERMAL_TRANSFORM_PATH}.")

    def _project_thermal_to_rgb(self, det: NormalizedDetection) -> NormalizedDetection:
        if self._transform is not None:
            pts = np.array([[det.x1, det.y1], [det.x2, det.y2]], dtype=np.float32).reshape(-1, 1, 2)
            proj = cv2.transform(pts, self._transform).reshape(-1, 2)
            (x1, y1), (x2, y2) = proj.tolist()
        else:
            x1, y1 = det.x1 + config.THERMAL_OFFSET_X, det.y1 + config.THERMAL_OFFSET_Y
            x2, y2 = det.x2 + config.THERMAL_OFFSET_X, det.y2 + config.THERMAL_OFFSET_Y
        return NormalizedDetection(x1, y1, x2, y2, det.fused_conf, det.cls_name, det.source)

    def fuse(
        self,
        rgb_dets: List[NormalizedDetection],
        thermal_dets: List[NormalizedDetection],
    ) -> List[NormalizedDetection]:
        """Merge simultaneous RGB + thermal detections into one list, one
        entry per physical object. Runs every frame now that both cameras
        are always on."""
        thermal_proj = [self._project_thermal_to_rgb(d) for d in thermal_dets]
        matched_thermal_idx = set()
        fused: List[NormalizedDetection] = []

        for r in rgb_dets:
            best_j, best_iou = -1, config.FUSION_IOU_THRESH
            for j, t in enumerate(thermal_proj):
                if j in matched_thermal_idx:
                    continue
                iou = _iou(r, t)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_j == -1:
                # RGB-only detection this frame (e.g. thermal missed a
                # low-heat-contrast object, or it's a bird/plane that
                # thermal doesn't distinguish). Still forwarded so the
                # tracker/priority queue can see it.
                fused.append(r)
                continue

            t = thermal_proj[best_j]
            matched_thermal_idx.add(best_j)
            w_r, w_t = config.FUSION_RGB_WEIGHT, config.FUSION_THERMAL_WEIGHT

            x1 = w_r * r.x1 + w_t * t.x1
            y1 = w_r * r.y1 + w_t * t.y1
            x2 = w_r * r.x2 + w_t * t.x2
            y2 = w_r * r.y2 + w_t * t.y2

            conf = min(1.0, w_r * r.fused_conf + w_t * t.fused_conf + config.FUSION_AGREEMENT_BONUS)

            # Thermal only ever reports the canonical "Drone" label (both
            # its raw classes are collapsed upstream in
            # detection_source.py). If thermal agrees with an RGB box,
            # trust the thermal heat signature over an RGB "Bird"/
            # "AirPlane" guess -- a live rotor signature co-located with
            # an RGB contact is a strong drone cue.
            cls_name = "Drone" if t.cls_name == "Drone" else r.cls_name

            fused.append(NormalizedDetection(x1, y1, x2, y2, conf, cls_name, "fused"))

        # Thermal detections that had no RGB match (e.g. RGB camera
        # blinded by glare/darkness, or object outside RGB's slightly
        # narrower FOV) are still forwarded on their own.
        for j, t in enumerate(thermal_proj):
            if j not in matched_thermal_idx:
                fused.append(t)

        return fused
