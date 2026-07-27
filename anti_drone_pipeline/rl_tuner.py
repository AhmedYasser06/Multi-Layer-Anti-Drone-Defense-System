"""
Layer 3 adaptive gain tuning via a small tabular Q-learning agent.

Why: a single hand-tuned PID (see pantilt_controller.py) is always a
compromise -- gains gentle enough to stay smooth on a slow, drifting
target will lag behind a fast one, and gains tight enough for a fast
target tend to overshoot/oscillate on a slow one. Rather than
hand-picking one fixed set of gains for the whole demo, this module
treats "should Kp go up, down, or stay" as a tiny reinforcement-learning
problem: every RL_UPDATE_INTERVAL_FRAMES it looks at how the tracking
error has been trending, chooses an action from a Q-table (epsilon-greedy
exploration), applies it to PanTiltController's proportional gains, and
rewards itself by how much the error actually improved.

Deliberately NOT deep RL -- there's no offline training run, no GPU, no
neural network. The state space is a handful of discretized
(error-magnitude, error-trend) buckets, so a plain Q-table converges
within the same live demo session, and the Q-table can optionally persist
to disk (RL_QTABLE_PATH) so it keeps improving across runs instead of
re-learning from scratch every time.

Scope: this ONLY scales how aggressively the gimbal chases a target that
the tracker has already selected. It never touches which target is
engaged -- that stays fully deterministic (NMSE priority scoring, see
priority.py) so engagement decisions remain auditable regardless of what
the RL agent has learned.
"""

import json
import os
import random
from typing import Tuple

from . import config


# Discretized error-magnitude buckets (pixels). Index is used as the
# "how far off are we" half of the state.
_ERROR_BUCKETS = (10, 30, 70, 150)  # boundaries; anything above the last -> bucket 4

# Actions: multiplicative-ish nudge applied to Kp (both axes together, so
# the table stays tiny). "hold" lets the agent do nothing once gains have
# converged on a good value for the current target behaviour.
_ACTIONS = ("increase", "decrease", "hold")


def _error_bucket(err_px: float) -> int:
    for i, edge in enumerate(_ERROR_BUCKETS):
        if err_px < edge:
            return i
    return len(_ERROR_BUCKETS)


def _trend_bucket(delta: float) -> str:
    if delta < -2.0:
        return "improving"
    if delta > 2.0:
        return "worsening"
    return "steady"


class RLGainTuner:
    """Wraps a PanTiltController and periodically nudges its PID
    proportional gains based on recent tracking-error trend."""

    def __init__(self, controller):
        self.controller = controller
        self.epsilon = config.RL_EPSILON_START
        self._q: dict = {}
        self._load_qtable()

        self._recent_errors = []
        self._last_state: Tuple[int, str] | None = None
        self._last_action: str | None = None
        self._last_error_at_update: float | None = None

    # ---------------- Q-table persistence ----------------
    def _load_qtable(self):
        path = config.RL_QTABLE_PATH
        if os.path.exists(path):
            try:
                with open(path) as f:
                    raw = json.load(f)
                # JSON keys are strings; state keys were saved as "bucket|trend"
                self._q = {}
                for key, actions in raw.items():
                    bucket_str, trend = key.split("|", 1)
                    self._q[(int(bucket_str), trend)] = actions
                print(f"[RL] Loaded gain-tuner Q-table from {path} ({len(self._q)} states).")
            except (OSError, ValueError) as e:
                print(f"[RL] Could not load Q-table ({e}); starting fresh.")
                self._q = {}

    def save_qtable(self):
        path = config.RL_QTABLE_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serializable = {f"{b}|{t}": actions for (b, t), actions in self._q.items()}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    # ---------------- Q-table access helpers ----------------
    def _q_row(self, state: Tuple[int, str]) -> dict:
        return self._q.setdefault(state, {a: 0.0 for a in _ACTIONS})

    def _choose_action(self, state: Tuple[int, str]) -> str:
        if random.random() < self.epsilon:
            return random.choice(_ACTIONS)
        row = self._q_row(state)
        return max(row, key=row.get)

    def _apply_action(self, action: str):
        step = config.RL_GAIN_STEP
        if action == "hold":
            return
        sign = 1.0 if action == "increase" else -1.0
        c = self.controller
        c.kp_x = max(config.RL_KP_MIN, min(config.RL_KP_MAX, c.kp_x + sign * step))
        c.kp_y = max(config.RL_KP_MIN, min(config.RL_KP_MAX, c.kp_y + sign * step))

    # ---------------- main entry point, called every ENGAGING frame ----------------
    def observe(self, pixel_error: float):
        """Feed the current frame's pixel error in. Internally batches
        RL_UPDATE_INTERVAL_FRAMES samples before doing a Q-learning update,
        so a single noisy frame doesn't cause a gain change."""
        if not config.RL_GAIN_TUNING_ENABLED:
            return

        self._recent_errors.append(pixel_error)
        if len(self._recent_errors) < config.RL_UPDATE_INTERVAL_FRAMES:
            return

        avg_error = sum(self._recent_errors) / len(self._recent_errors)
        self._recent_errors = []

        if self._last_state is not None and self._last_error_at_update is not None:
            # Reward = how much the average error dropped since the last
            # update (positive = improvement). Reinforces gain choices
            # that actually reduce tracking error, penalizes ones that
            # make it worse.
            reward = self._last_error_at_update - avg_error
            self._update_q(self._last_state, self._last_action, reward, avg_error)

        trend = _trend_bucket(avg_error - (self._last_error_at_update or avg_error))
        state = (_error_bucket(avg_error), trend)
        action = self._choose_action(state)
        self._apply_action(action)

        self._last_state = state
        self._last_action = action
        self._last_error_at_update = avg_error
        self.epsilon = max(config.RL_EPSILON_MIN, self.epsilon * config.RL_EPSILON_DECAY)

    def _update_q(self, state, action, reward, avg_error_now):
        next_state = (_error_bucket(avg_error_now), _trend_bucket(0.0))
        row = self._q_row(state)
        next_row = self._q_row(next_state)
        best_next = max(next_row.values())
        row[action] += config.RL_LEARNING_RATE * (
            reward + config.RL_DISCOUNT * best_next - row[action]
        )

    def reset_episode(self):
        """Call when a target is lost / engagement ends, so the reward
        signal from one target's tracking behaviour doesn't get attributed
        to the next, unrelated target."""
        self._recent_errors = []
        self._last_state = None
        self._last_action = None
        self._last_error_at_update = None
