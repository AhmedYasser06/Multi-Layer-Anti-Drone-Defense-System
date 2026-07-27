"""
Central configuration. Change these once instead of hunting through modules.
"""

# ---------------- Serial link to Arduino ----------------
SERIAL_PORT = "COM5"         
SERIAL_BAUD = 115200
SERIAL_TIMEOUT_S = 0.05

# ---------------- Cameras ----------------
RGB_CAM_INDEX = 0             # cv2.VideoCapture index for the RGB camera
THERMAL_CAM_INDEX = 1         # cv2.VideoCapture index for the thermal camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------------- Models ----------------
RGB_MODEL_PATH = "models/rgb_multiclass_yolov8s_best.pt"
THERMAL_MODEL_PATH = "models/thermal_yolov8_best.pt"
RGB_CONF_THRESH = 0.35
THERMAL_CONF_THRESH = 0.35
DEVICE = "cpu"                # "cuda:0"

# RGB model has 4 classes; only "Drone" is treated as a threat for
# priority/engagement. Bird/AirPlane/Helicopter are still detected+drawn
# but excluded from the priority queue.
# Thermal model's 2 raw classes are BOTH collapsed to "Drone" in
# detection_source.py before they ever reach this point -- so at this
# layer, "threat" always just means the single canonical name below.
THREAT_CLASSES = {"Drone"}
NON_THREAT_CLASSES = {"Bird", "AirPlane", "Helicopter"}

# Class-agnostic NMS threshold used to dedup thermal's 2-class overlap
# (see detection_source.py:_dedup_same_object)
THERMAL_DEDUP_IOU = 0.5

# ---------------- Sensor fusion (RGB + Thermal, simultaneous) ----------------
# Both cameras now run every frame. 
# fusion.py projects thermal boxes into RGB pixel space and
# merges any RGB+thermal pair that refers to the same physical object.
#
# Preferred: a calibrated similarity transform produced by
# tools/calibrate_thermal_offset.py, saved here. If that file doesn't
# exist yet (rig not calibrated), fusion.py falls back to a constant
# (dx, dy) pixel offset below -- good enough for a rough demo, but you
# should calibrate for real alignment, especially near frame edges.
THERMAL_TRANSFORM_PATH = "models/thermal_to_rgb_transform.json"
THERMAL_OFFSET_X = 0.0
THERMAL_OFFSET_Y = 0.0

# IoU floor (in RGB pixel space, after projecting thermal boxes) to treat
# an RGB box and a thermal box as the same physical target.
FUSION_IOU_THRESH = 0.25

# Confidence/position fusion weights when RGB and thermal both fire on
# the same target. Thermal gets slightly more weight by default since a
# hot rotor/motor signature is a strong drone cue and is less confusable
# with birds than RGB shape alone; tune based on your own test data.
FUSION_RGB_WEIGHT = 0.45
FUSION_THERMAL_WEIGHT = 0.55
# Small confidence bonus applied when both sensors agree on the same
# target (rewards cross-sensor agreement without letting it saturate to
# 1.0 for a single weak pair of detections).
FUSION_AGREEMENT_BONUS = 0.05

# ---------------- Radar (Layer 2) ----------------
RADAR_MAX_RANGE_CM = 400.0    # HC-SR04 practical max; raise once you swap to mmWave
RADAR_TRIGGER_RANGE_CM = 350.0  # distance below which we treat it as "something is there"
RADAR_ANGLE_TO_PAN_SCALE = 1.0  # 1.0 if radar servo and pan servo share the same 0-180 frame
RADAR_ANGLE_TO_PAN_OFFSET = 0.0

# ---------------- Pan-tilt visual servoing (Layer 3) ----------------
# Full PID now (report Section 11 future-work item: "PID Controller for
# Pan-Tilt" -- upgraded from the pure-proportional controller). Integral
# term removes steady-state lag on a steadily-drifting target; derivative
# term damps overshoot when the proportional term alone would oscillate.
PAN_TILT_KP_X = 0.03           # proportional gain, pixel error -> degrees
PAN_TILT_KI_X = 0.0015         # integral gain -- corrects persistent offset
PAN_TILT_KD_X = 0.012          # derivative gain -- damps oscillation/overshoot
PAN_TILT_KP_Y = 0.03
PAN_TILT_KI_Y = 0.0015
PAN_TILT_KD_Y = 0.012
PID_INTEGRAL_CLAMP = 400.0      # anti-windup clamp on the accumulated integral term
PAN_TILT_MAX_STEP_DEG = 4       # clamp per-frame servo movement (avoid jitter/overshoot)
PAN_CENTER_DEG = 90
TILT_CENTER_DEG = 90

# ---------------- RL-based adaptive gain tuning (Layer 3, optional) ----------------
# Report Section 11 lists the PID upgrade as future work; this goes one
# step further with a small tabular Q-learning agent that nudges Kp up or
# down at runtime based on whether recent tracking error is trending
# down. It only ever scales the existing PID gains -- it does not touch
# targeting/priority decisions, which stay fully deterministic (NMSE,
# see priority.py) so engagement choices remain auditable.
RL_GAIN_TUNING_ENABLED = True
RL_UPDATE_INTERVAL_FRAMES = 10   # observe this many frames of error before an update
RL_LEARNING_RATE = 0.1
RL_DISCOUNT = 0.9
RL_EPSILON_START = 0.3           # initial exploration rate
RL_EPSILON_MIN = 0.05
RL_EPSILON_DECAY = 0.995         # multiplied in after every update
RL_GAIN_STEP = 0.004             # how much one RL action changes Kp
RL_KP_MIN = 0.008
RL_KP_MAX = 0.09
RL_QTABLE_PATH = "logs/rl_gain_qtable.json"  # persisted across runs so it keeps improving

# ---------------- Target lock (Layer 3/4) ----------------
LOCK_PIXEL_RADIUS = 15
LOCK_FRAMES_REQUIRED = 20       # ~0.4s @ 50Hz control loop matches the report's spec
TRACK_LOST_FRAMES = 10          # frames with no detection before dropping a track

# ---------------- NMSE priority scoring (Layer 4) ----------------
# Weights must sum to 1.0. Tune based on which factor matters most for your demo.
PRIORITY_WEIGHTS = {
    "distance_to_center": 0.30,   # closer to protected-zone center = higher threat
    "closing_speed":      0.30,   # moving toward the zone = higher threat
    "confidence":         0.25,   # fused YOLO confidence
    "bbox_size":          0.15,   # bigger box = closer/more imminent
}
assert abs(sum(PRIORITY_WEIGHTS.values()) - 1.0) < 1e-6

# Frame-normalization reference for bbox_size scoring (fraction of frame area
# at which we consider the target "as close as it gets")
BBOX_AREA_SATURATION_FRACTION = 0.25

# ---------------- Logging ----------------
LOG_DIR = "logs"
