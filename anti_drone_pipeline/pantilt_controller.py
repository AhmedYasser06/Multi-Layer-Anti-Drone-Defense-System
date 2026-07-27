"""
Layer 3 fine pointing: converts pixel error (target center vs frame center)
into incremental pan/tilt servo commands.

Full PID now (report Section 11 future-work item: "PID Controller for
Pan-Tilt", upgraded from the pure-proportional controller in the original
prototype):
  - P term: reacts to the current pixel error, same as before.
  - I term: accumulates error over time so a target that sits with a
    small but persistent offset (e.g. a slow steady drift the P term
    alone never fully closes) still gets pulled to center instead of
    settling with steady-state lag. Clamped (anti-windup) so a long
    period with no target doesn't leave a huge accumulated term that
    then whips the gimbal once a target reappears.
  - D term: reacts to how fast the error is changing, damping the
    overshoot/oscillation a P-only or PI controller tends to produce on
    a fast-moving target.

Gains (self.kp_x/kp_y etc.) are plain instance attributes, not just
locals read from config each call -- this lets rl_tuner.py adjust them
at runtime (see RL_GAIN_TUNING_ENABLED in config.py) without touching
this file.
"""

import time

from . import config


class PanTiltController:
    def __init__(self, serial_link):
        self.link = serial_link
        self.pan = config.PAN_CENTER_DEG
        self.tilt = config.TILT_CENTER_DEG
        self.link.send_pan_tilt(self.pan, self.tilt)

        # PID gains -- mutable at runtime (see rl_tuner.py).
        self.kp_x, self.ki_x, self.kd_x = config.PAN_TILT_KP_X, config.PAN_TILT_KI_X, config.PAN_TILT_KD_X
        self.kp_y, self.ki_y, self.kd_y = config.PAN_TILT_KP_Y, config.PAN_TILT_KI_Y, config.PAN_TILT_KD_Y

        # Running PID state.
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._prev_err_x = 0.0
        self._prev_err_y = 0.0
        self._prev_t = None

    def slew_to_radar_angle(self, radar_angle_deg: float):
        """Coarse jump (Layer 2 -> Layer 3 handoff): point the gimbal at the
        angle the radar sweep reported, then let update() take over for
        fine visual tracking."""
        pan = radar_angle_deg * config.RADAR_ANGLE_TO_PAN_SCALE
        pan += config.RADAR_ANGLE_TO_PAN_OFFSET
        pan = max(0, min(180, pan))
        self.pan = pan
        self.tilt = config.TILT_CENTER_DEG
        self.link.send_pan_tilt(self.pan, self.tilt)
        self.reset_pid()

    def reset_pid(self):
        """Clear accumulated integral/derivative state. Call this on every
        SLEWING->ENGAGING handoff and on target-loss so stale error from a
        previous, unrelated target never leaks into a new engagement."""
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._prev_err_x = 0.0
        self._prev_err_y = 0.0
        self._prev_t = None

    def update(self, target_cx, target_cy, frame_w, frame_h) -> float:
        """Full PID correction toward the tracked target. Returns the
        pixel distance from frame center (used by the lock-detection logic
        in main.py)."""
        frame_cx, frame_cy = frame_w / 2.0, frame_h / 2.0
        err_x = target_cx - frame_cx
        err_y = target_cy - frame_cy  # image y grows downward

        now = time.time()
        dt = max(1e-3, now - self._prev_t) if self._prev_t is not None else 1e-2
        self._prev_t = now

        # Integral, with anti-windup clamp.
        self._integral_x = max(-config.PID_INTEGRAL_CLAMP,
                                min(config.PID_INTEGRAL_CLAMP, self._integral_x + err_x * dt))
        self._integral_y = max(-config.PID_INTEGRAL_CLAMP,
                                min(config.PID_INTEGRAL_CLAMP, self._integral_y + err_y * dt))

        # Derivative.
        deriv_x = (err_x - self._prev_err_x) / dt
        deriv_y = (err_y - self._prev_err_y) / dt
        self._prev_err_x, self._prev_err_y = err_x, err_y

        pan_delta = (self.kp_x * err_x) + (self.ki_x * self._integral_x) + (self.kd_x * deriv_x)
        # tilt up is usually +, image y grows downward -> negate
        tilt_delta = -((self.kp_y * err_y) + (self.ki_y * self._integral_y) + (self.kd_y * deriv_y))

        pan_delta = max(-config.PAN_TILT_MAX_STEP_DEG, min(config.PAN_TILT_MAX_STEP_DEG, pan_delta))
        tilt_delta = max(-config.PAN_TILT_MAX_STEP_DEG, min(config.PAN_TILT_MAX_STEP_DEG, tilt_delta))

        self.pan = max(0, min(180, self.pan + pan_delta))
        self.tilt = max(20, min(160, self.tilt + tilt_delta))
        self.link.send_pan_tilt(self.pan, self.tilt)

        return (err_x ** 2 + err_y ** 2) ** 0.5

    def recenter(self):
        self.pan, self.tilt = config.PAN_CENTER_DEG, config.TILT_CENTER_DEG
        self.link.send_pan_tilt(self.pan, self.tilt)
        self.reset_pid()
