"""Speed tracking for the scripted vehicles (lead profile, scripted ego driver).

Lab3's cruise control, made smooth: a reference speed is integrated from the desired
acceleration and tracked with a PI controller on throttle / brake. Returns plain numbers;
the CARLA runner turns them into carla.VehicleControl.
"""

import numpy as np


class SpeedController:
    def __init__(self, kp=0.35, ki=0.08, throttle_ff=0.25, brake_gain=0.25, max_throttle=0.85):
        self.kp, self.ki = kp, ki
        self.throttle_ff = throttle_ff
        self.brake_gain = brake_gain
        self.max_throttle = max_throttle
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def step(self, dt, speed_mps, target_mps):
        """Return (throttle, brake) in [0, 1] to reach target_mps."""
        if target_mps <= 0.05 and speed_mps < 0.3:
            self.integral = 0.0
            return 0.0, 0.8                     # standstill: hold the brake
        error = target_mps - speed_mps
        self.integral = float(np.clip(self.integral + error * dt, -2.0, 2.0))
        u = self.kp * error + self.ki * self.integral
        if u >= -0.05:
            ff = self.throttle_ff if target_mps > 0.05 else 0.0
            return float(np.clip(ff + u, 0.0, self.max_throttle)), 0.0
        return 0.0, float(np.clip(-u * self.brake_gain / 0.35, 0.0, 1.0))


class AccelToSpeed:
    """Integrates a desired acceleration into a reference speed (for drivers that output accelerations)."""

    def __init__(self, initial_speed=0.0):
        self.reference = initial_speed

    def step(self, dt, accel, actual_speed):
        # Do not let the reference run away from reality (e.g. while the brake is held).
        self.reference = float(np.clip(self.reference + accel * dt, 0.0, actual_speed + 2.0))
        if accel < 0:
            self.reference = min(self.reference, actual_speed + accel * dt)
        return max(0.0, self.reference)
