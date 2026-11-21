"""Constant-velocity Kalman filter for the gap to the lead vehicle.

State x = [gap, gap_rate]: gap in meters (bumper to bumper), gap_rate in m/s
(negative = closing in, the opposite of the closing speed).

    predict:  x <- F x,            P <- F P F^T + Q       F = [[1, dt], [0, 1]]
    update:   y = z - H x,         S = H P H^T + R
              K = P H^T S^-1,      x <- x + K y,   P <- (I - K H) P

Q is the white-noise-acceleration model: the gap rate changes by random accelerations of
standard deviation sigma_accel (the lead braking / accelerating in stop-and-go traffic).

Sensors plug in through H and R:
- radar  z = [gap, gap_rate], H = I            (range and Doppler range-rate)
- camera z = [gap],           H = [[1, 0]]     (monocular distance only)
"""

import numpy as np

H_RADAR = np.eye(2)
H_CAMERA = np.array([[1.0, 0.0]])


def process_noise(dt, sigma_accel):
    """Discrete white-noise-acceleration Q for the constant-velocity model."""
    return sigma_accel ** 2 * np.array([
        [dt ** 4 / 4.0, dt ** 3 / 2.0],
        [dt ** 3 / 2.0, dt ** 2],
    ])


class ConstantVelocityKF:
    def __init__(self, x0, P0, sigma_accel):
        self.x = np.asarray(x0, dtype=float).reshape(2)
        self.P = np.asarray(P0, dtype=float).reshape(2, 2)
        self.sigma_accel = sigma_accel

    @property
    def gap(self):
        return float(self.x[0])

    @property
    def gap_rate(self):
        return float(self.x[1])

    @property
    def sigma_gap(self):
        return float(np.sqrt(self.P[0, 0]))

    @property
    def sigma_rate(self):
        return float(np.sqrt(self.P[1, 1]))

    def predict(self, dt):
        F = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + process_noise(dt, self.sigma_accel)

    def innovation(self, z, H, R):
        """(y, S, NIS) of a measurement against the current prediction, without updating."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        nis = float(y @ np.linalg.solve(S, y))
        return y, S, nis

    def update(self, z, H, R):
        """Standard update; returns the normalized innovation squared (NIS) of the measurement."""
        y, S, nis = self.innovation(z, H, R)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        # Joseph form: stays symmetric positive-definite with finite precision.
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return nis
