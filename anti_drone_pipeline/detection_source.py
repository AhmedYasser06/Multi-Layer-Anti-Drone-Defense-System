"""
Simultaneous RGB + thermal detection.

BOTH cameras and BOTH models run every frame now -- there is no day/night
mode that picks a single active sensor:
  RGB     -> RGB camera + RGB multiclass model (Bird / Drone / AirPlane / Helicopter)
  Thermal -> Thermal camera + thermal model (2 raw classes, BOTH forced to "Drone")

The two detection lists are handed to fusion.SensorFusion (see fusion.py)
every frame, which projects thermal boxes into RGB pixel space and merges
any pair that refers to the same physical object. What still matters
here, per-sensor, before fusion happens:

  - RGB: only the "Drone" class is a threat. Bird/AirPlane/Helicopter are
    still detected and drawn (useful to prove to judges the system isn't
    just laser-locking anything that moves) but excluded from the
    priority queue via config.THREAT_CLASSES.

  - Thermal: whatever its 2 raw class labels actually are (drone/bird,
    hot-object/cold-object, whatever the dataset used), BOTH get
    collapsed to the canonical "Drone" label before reaching fusion --
    per your instruction to treat both thermal classes as drone. If the
    thermal model fires two overlapping boxes on the same physical object
    (one from each class), a dedup pass keeps only the higher-confidence
    box so fusion doesn't see duplicate thermal boxes for one target.
"""

from dataclasses import dataclass
from typing import List

from ultralytics import YOLO

from . import config


@dataclass
class NormalizedDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    fused_conf: float  # named to match what tracker.py expects; no fusion math applied
    cls_name: str
    source: str  # "rgb" or "thermal"

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self):
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


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


def _dedup_same_object(
    dets: List[NormalizedDetection], iou_thresh: float
) -> List[NormalizedDetection]:
    """Class-agnostic NMS. Used after forcing both thermal classes to
    'Drone' so two overlapping same-object detections don't become two
    separate tracks."""
    dets = sorted(dets, key=lambda d: d.fused_conf, reverse=True)
    kept: List[NormalizedDetection] = []
    for d in dets:
        if all(_iou(d, k) < iou_thresh for k in kept):
            kept.append(d)
    return kept


class FusedDetector:
    def __init__(self):
        self.rgb_model = YOLO(config.RGB_MODEL_PATH)
        self.thermal_model = YOLO(config.THERMAL_MODEL_PATH)
        # Local import to avoid a circular import at module load time
        # (fusion.py imports NormalizedDetection from this module).
        from .fusion import SensorFusion
        self._fusion = SensorFusion()

    def infer_rgb(self, frame) -> List[NormalizedDetection]:
        results = self.rgb_model.predict(
            frame, conf=config.RGB_CONF_THRESH, device=config.DEVICE, verbose=False
        )[0]
        out = []
        if results.boxes is None:
            return out
        names = results.names
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            out.append(NormalizedDetection(x1, y1, x2, y2, conf, names[cls_id], "rgb"))
        return out

    def infer_thermal(self, frame) -> List[NormalizedDetection]:
        results = self.thermal_model.predict(
            frame, conf=config.THERMAL_CONF_THRESH, device=config.DEVICE, verbose=False
        )[0]
        raw = []
        if results.boxes is None:
            return raw
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            # Collapse both thermal classes to the canonical threat label,
            # regardless of which of the 2 classes actually fired.
            raw.append(NormalizedDetection(x1, y1, x2, y2, conf, "Drone", "thermal"))
        return _dedup_same_object(raw, iou_thresh=config.THERMAL_DEDUP_IOU)

    def infer_fused(self, rgb_frame, thermal_frame) -> List[NormalizedDetection]:
        """Runs both models every frame and fuses the results. Replaces
        the old day/night infer_active(mode, frame) single-source path --
        both cameras are always open and always inferred now."""
        rgb_dets = self.infer_rgb(rgb_frame)
        thermal_dets = self.infer_thermal(thermal_frame)
        return self._fusion.fuse(rgb_dets, thermal_dets)
