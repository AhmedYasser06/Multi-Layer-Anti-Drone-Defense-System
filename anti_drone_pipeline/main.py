"""
Full pipeline entry point. Run with: python -m anti_drone_pipeline.main

State machine:
  SCANNING   -> Arduino radar sweep active.
  SLEWING    -> An object crossed RADAR_TRIGGER_RANGE_CM; gimbal jumps to
                that angle and radar sweep is paused.
  ENGAGING   -> RGB and thermal cameras are BOTH read every frame (no more
                day/night single-source switching); their detections are
                fused (see fusion.py) before reaching the tracker. Tracker
                + NMSE priority queue pick the top threat, PID pan-tilt
                (optionally RL-gain-tuned) servos onto it.
  LOCKED     -> Target held within LOCK_PIXEL_RADIUS for LOCK_FRAMES_REQUIRED
                consecutive frames -> laser fires, event logged.
  (loses target for TRACK_LOST_FRAMES -> back to SCANNING, radar resumes)
"""

import time

import cv2

from . import config
from .serial_link import SerialLink
from .detection_source import FusedDetector
from .tracker import CentroidIoUTracker
from .priority import PriorityQueue
from .pantilt_controller import PanTiltController
from .rl_tuner import RLGainTuner
from .logger import EventLogger

STATE_SCANNING = "SCANNING"
STATE_SLEWING = "SLEWING"
STATE_ENGAGING = "ENGAGING"
STATE_LOCKED = "LOCKED"


def open_camera(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    return cap


def main():
    print("Connecting to Arduino...")
    link = SerialLink()
    if not link.ping():
        print("WARNING: Arduino did not respond to PING. Check port/wiring.")
    else:
        print("Arduino OK.")

    print("Loading models (this can take a few seconds)...")
    detector = FusedDetector()

    # Both cameras are opened up front and stay open for the whole run --
    # no more day/night logic that only opens/reads one of them at a time.
    rgb_cap = open_camera(config.RGB_CAM_INDEX)
    thermal_cap = open_camera(config.THERMAL_CAM_INDEX)

    tracker = CentroidIoUTracker()
    pq = PriorityQueue()
    gimbal = PanTiltController(link)
    rl_tuner = RLGainTuner(gimbal)
    logger = EventLogger()

    state = STATE_SCANNING
    link.set_radar_sweeping(True)
    link.set_jammer(True)  # Layer 1: jammer active continuously by default
    logger.log("SYSTEM_START", detail="Layer1 jammer engaged, radar sweep active, RGB+thermal fusion active")

    engaged_track_id = None
    lock_streak = 0
    frames_since_detection = 0

    try:
        while True:
            # ---------------- STATE: SCANNING ----------------
            if state == STATE_SCANNING:
                sample = link.get_radar_sample(block=False)
                if sample is not None:
                    angle, dist_cm = sample
                    if dist_cm <= config.RADAR_TRIGGER_RANGE_CM:
                        print(f"[RADAR] Object at angle={angle} dist={dist_cm}cm -> slewing")
                        logger.log("RADAR_CONTACT", detail=f"angle={angle}, dist={dist_cm}")
                        link.set_radar_sweeping(False)
                        gimbal.slew_to_radar_angle(angle)
                        state = STATE_SLEWING
                        slew_start_t = time.time()
                else:
                    time.sleep(0.01)
                continue

            # ---------------- STATE: SLEWING ----------------
            if state == STATE_SLEWING:
                if time.time() - slew_start_t > 0.4:
                    state = STATE_ENGAGING
                    frames_since_detection = 0
                continue

            # ---------------- STATE: ENGAGING / LOCKED ----------------
            # Both cameras are read every frame now.
            ok_rgb, rgb_frame = rgb_cap.read()
            ok_thermal, thermal_frame = thermal_cap.read()
            if not ok_rgb or not ok_thermal:
                continue

            dets = detector.infer_fused(rgb_frame, thermal_frame)
            tracks = tracker.update(dets)

            pq.rebuild(tracks, config.FRAME_WIDTH, config.FRAME_HEIGHT)
            top = pq.top_target()

            # Display frame: RGB with thermal picture-in-picture, so both
            # feeds are visible at once instead of only whichever camera
            # used to be "active".
            display_frame = rgb_frame.copy()
            thumb = cv2.resize(thermal_frame, (config.FRAME_WIDTH // 4, config.FRAME_HEIGHT // 4))
            display_frame[8:8 + thumb.shape[0], 8:8 + thumb.shape[1]] = thumb

            if top is None:
                frames_since_detection += 1
                if frames_since_detection > config.TRACK_LOST_FRAMES:
                    print("[TRACK] Target lost -> resuming radar scan")
                    logger.log("TRACK_LOST", track_id=engaged_track_id)
                    link.set_laser(False)
                    engaged_track_id = None
                    lock_streak = 0
                    rl_tuner.reset_episode()
                    state = STATE_SCANNING
                    link.set_radar_sweeping(True)
                cv2.putText(display_frame, f"STATE:{state} (searching, RGB+Thermal fused)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("Anti-Drone View", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            frames_since_detection = 0
            if top.track_id != engaged_track_id:
                print(f"[PRIORITY] Switching engagement to track {top.track_id} "
                      f"({top.cls_name}, score={top.score:.2f})")
                logger.log("PRIORITY_SWITCH", track_id=top.track_id,
                           cls_name=top.cls_name, score=round(top.score, 3))
                engaged_track_id = top.track_id
                lock_streak = 0
                link.set_laser(False)
                rl_tuner.reset_episode()
                state = STATE_ENGAGING

            tr = tracks[top.track_id]
            pixel_err = gimbal.update(tr.cx, tr.cy, config.FRAME_WIDTH, config.FRAME_HEIGHT)
            rl_tuner.observe(pixel_err)  # Layer 3 adaptive PID gain tuning

            if pixel_err <= config.LOCK_PIXEL_RADIUS:
                lock_streak += 1
            else:
                lock_streak = 0

            if lock_streak >= config.LOCK_FRAMES_REQUIRED and state != STATE_LOCKED:
                state = STATE_LOCKED
                link.set_laser(True)
                print(f"[LOCK] Target {top.track_id} ({top.cls_name}) LOCKED. Laser ON.")
                logger.log("TARGET_LOCK", track_id=top.track_id,
                           cls_name=top.cls_name, score=round(top.score, 3))
            elif lock_streak < config.LOCK_FRAMES_REQUIRED and state == STATE_LOCKED:
                state = STATE_ENGAGING
                link.set_laser(False)

            # Optional live view for debugging/demo -- comment out for headless ops.
            for d in dets:
                color = (0, 255, 0) if d.cls_name in config.THREAT_CLASSES else (0, 165, 255)
                cv2.rectangle(display_frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2)
                cv2.putText(display_frame, f"{d.cls_name}({d.source}) {d.fused_conf:.2f}",
                            (int(d.x1), int(d.y1) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)
            cv2.putText(display_frame, f"STATE:{state}  Kp=({gimbal.kp_x:.4f},{gimbal.kp_y:.4f})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Anti-Drone View", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        link.set_laser(False)
        link.set_jammer(False)
        link.set_radar_sweeping(True)
        rgb_cap.release()
        thermal_cap.release()
        cv2.destroyAllWindows()
        logger.close()
        link.close()
        rl_tuner.save_qtable()


if __name__ == "__main__":
    main()
