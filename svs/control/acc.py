"""Gentle slow-down: a basic ACC-style longitudinal controller.

    desired gap   d* = standstill_gap + time_gap * v_ego
    acceleration  a  = k_gap * (gap - d*) + k_rate * gap_rate        (lead tracked)
                  a  = min(a, k_speed * (v_cruise - v_ego))          (never exceed cruise speed)

limited to [-comfort_decel, max_accel] and by a jerk limit, so the slow-down is gentle.
Only when TTC drops below emergency_ttc_s may it brake harder (up to emergency_decel),
without the jerk limit. The acceleration is turned into throttle / brake with a simple
linear map (Lab3 cruise control did the same with fixed steps).
"""

from dataclasses import dataclass

import numpy as np

from svs.decision.warning import time_to_collision


@dataclass(frozen=True)
class ControlCommand:
    accel_mps2: float
    throttle: float
    brake: float
    reason: str   # "cruise", "follow", "emergency", "hold"


def accel_to_pedals(accel, cfg):
    """Desired acceleration -> (throttle, brake) in [0, 1]."""
    if accel >= -0.2:
        # Small decelerations come from rolling resistance with the throttle released.
        throttle = cfg.throttle_bias + cfg.throttle_per_mps2 * accel if accel >= 0 else 0.0
        return float(np.clip(throttle, 0.0, 1.0)), 0.0
    return 0.0, float(np.clip(-accel * cfg.brake_per_mps2, 0.0, 1.0))


class GapController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.accel = 0.0

    def reset(self):
        self.accel = 0.0

    def desired_gap(self, ego_speed_mps):
        return self.cfg.standstill_gap_m + self.cfg.time_gap_s * ego_speed_mps

    def step(self, dt, estimate, ego_speed_mps):
        cfg = self.cfg
        cruise = cfg.k_speed * (cfg.cruise_speed_mps - ego_speed_mps)
        lower, reason, emergency = -cfg.comfort_decel_mps2, "cruise", False

        if estimate is not None and estimate.confirmed:
            rate = estimate.gap_rate_mps or 0.0
            follow = cfg.k_gap * (estimate.gap_m - self.desired_gap(ego_speed_mps)) + cfg.k_rate * rate
            target = min(follow, cruise)
            reason = "follow" if follow < cruise else "cruise"
            if time_to_collision(estimate.gap_m, -rate) < cfg.emergency_ttc_s:
                lower, reason, emergency = -cfg.emergency_decel_mps2, "emergency", True
        else:
            target = cruise

        target = float(np.clip(target, lower, cfg.max_accel_mps2))
        if not emergency:
            step = cfg.max_jerk_mps3 * dt
            target = float(np.clip(target, self.accel - step, self.accel + step))
        self.accel = target

        if ego_speed_mps < 0.3 and target <= 0.0:
            # Standing still behind the lead: hold the brake, as a driver would.
            self.accel = 0.0
            return ControlCommand(0.0, 0.0, 0.6, "hold")

        throttle, brake = accel_to_pedals(target, cfg)
        return ControlCommand(target, throttle, brake, reason)
