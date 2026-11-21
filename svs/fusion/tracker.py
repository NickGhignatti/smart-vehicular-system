"""Lead-vehicle track: Kalman filter + track management ("object persistence").

One `LeadTracker` per run. Every frame call `step(dt, camera, radar)` with the
GapMeasurements of that frame (None when a sensor has nothing) and read the TrackEstimate.

Modes, all on the same filter so the comparison is fair:
- "camera": camera updates only (closing speed must be inferred from successive gaps);
- "radar":  radar updates only (range + Doppler range-rate);
- "fused":  both, radar first (it carries the rate), then camera.
Plus two baselines without any filter: "camera_raw", "radar_raw" (RawEstimator).

Track life cycle:
    none --measurement--> tentative --M of N updates--> confirmed
    confirmed --no update--> coasting (prediction only) --too long / too uncertain--> none
A measurement far from the prediction (NIS above the chi-square gate) is rejected: it is
probably another object (ghost, adjacent car). Many consecutive rejections mean the lead
itself changed (cut-in, lead left the lane): the track restarts from the new measurements.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from svs.fusion.kalman import H_CAMERA, H_RADAR, ConstantVelocityKF

MODES = ("camera_raw", "radar_raw", "camera", "radar", "fused")


@dataclass(frozen=True)
class TrackEstimate:
    status: str                              # "none", "tentative", "confirmed", "coasting", "raw"
    gap_m: Optional[float] = None
    gap_rate_mps: Optional[float] = None     # negative = closing in
    sigma_gap_m: Optional[float] = None
    sigma_rate_mps: Optional[float] = None
    updated_by: tuple = ()                   # sources accepted this frame
    rejected: tuple = ()                     # sources gated out this frame
    nis: dict = field(default_factory=dict)  # source -> NIS of the accepted update

    @property
    def valid(self):
        return self.gap_m is not None

    @property
    def confirmed(self):
        """True when downstream logic (warning, controller) may act on the estimate."""
        return self.status in ("confirmed", "coasting", "raw") and self.valid

    @property
    def closing_speed_mps(self):
        return None if self.gap_rate_mps is None else -self.gap_rate_mps


def _model(m):
    """(z, H, R) of a GapMeasurement for the Kalman filter."""
    if m.rate_mps is not None:
        return (np.array([m.gap_m, m.rate_mps]), H_RADAR,
                np.diag([m.sigma_gap_m ** 2, m.sigma_rate_mps ** 2]))
    return np.array([m.gap_m]), H_CAMERA, np.array([[m.sigma_gap_m ** 2]])


class LeadTracker:
    def __init__(self, cfg, mode="fused"):
        if mode not in ("camera", "radar", "fused"):
            raise ValueError(f"LeadTracker mode must be camera, radar or fused, not {mode!r}")
        self.cfg = cfg
        self.mode = mode
        self.kf = None
        self.status = "none"
        self.history = deque(maxlen=cfg.confirm_window)
        self.since_update_s = 0.0
        self.consecutive_rejects = 0
        self.camera_hits = 0

    def reset(self):
        self.kf = None
        self.status = "none"
        self.history.clear()
        self.since_update_s = 0.0
        self.consecutive_rejects = 0
        self.camera_hits = 0

    # --- public ------------------------------------------------------------------------

    def step(self, dt, camera=None, radar=None):
        camera = camera if self.mode in ("camera", "fused") else None
        radar = radar if self.mode in ("radar", "fused") else None

        if self.kf is None:
            used = self._initialize(camera, radar)
            return self._estimate(used, (), {})

        self.kf.predict(dt)
        accepted, rejected, nis = [], [], {}
        for m in (radar, camera):
            if m is None:
                continue
            z, H, R = _model(m)
            _, _, value = self.kf.innovation(z, H, R)
            gate = self.cfg.gate_chi2_2d if len(z) == 2 else self.cfg.gate_chi2_1d
            if value <= gate:
                self.kf.update(z, H, R)
                accepted.append(m.source)
                nis[m.source] = value
            else:
                rejected.append(m.source)

        self._bookkeeping(dt, accepted, rejected)

        if self.consecutive_rejects >= self.cfg.reinit_after:
            # The measurements keep disagreeing with the track: the lead has changed.
            self.reset()
            used = self._initialize(camera, radar)
            return self._estimate(used, tuple(rejected), {})

        if self._should_drop():
            self.reset()
        return self._estimate(tuple(accepted), tuple(rejected), nis)

    # --- internals ---------------------------------------------------------------------

    def _initialize(self, camera, radar):
        """Start a tentative track; returns the sources used."""
        cfg = self.cfg
        if camera is not None and radar is not None:
            agree = (abs(camera.lateral_m - radar.lateral_m) <= cfg.max_lateral_disagreement_m
                     and abs(camera.gap_m - radar.gap_m)
                     <= 3.0 * np.hypot(camera.sigma_gap_m, radar.sigma_gap_m))
            if agree:
                self._start(radar)
                z, H, R = _model(camera)
                self.kf.update(z, H, R)
                self.camera_hits = 1
                return ("radar", "camera")
            # Disagreement: trust the camera, which knows the object is a vehicle in our lane.
            self._start(camera)
            self.camera_hits = 1
            return ("camera",)
        if radar is not None:
            self._start(radar)
            return ("radar",)
        if camera is not None:
            self._start(camera)
            self.camera_hits = 1
            return ("camera",)
        return ()

    def _start(self, m):
        if m.rate_mps is not None:
            x0 = [m.gap_m, m.rate_mps]
            P0 = np.diag([m.sigma_gap_m ** 2, m.sigma_rate_mps ** 2])
        else:
            x0 = [m.gap_m, 0.0]
            P0 = np.diag([m.sigma_gap_m ** 2, self.cfg.init_rate_sigma_mps ** 2])
        self.kf = ConstantVelocityKF(x0, P0, self.cfg.sigma_accel_mps2)
        self.status = "tentative"
        self.history.clear()
        self.history.append(True)
        self.since_update_s = 0.0
        self.consecutive_rejects = 0

    def _bookkeeping(self, dt, accepted, rejected):
        self.history.append(bool(accepted))
        if "camera" in accepted:
            self.camera_hits += 1
        if accepted:
            self.since_update_s = 0.0
            self.consecutive_rejects = 0
        else:
            self.since_update_s += dt
            self.consecutive_rejects = self.consecutive_rejects + 1 if rejected else 0

        if self.status == "tentative":
            needs_camera = self.mode == "fused" and self.cfg.fused_confirm_requires_camera
            if sum(self.history) >= self.cfg.confirm_hits and (self.camera_hits > 0 or not needs_camera):
                self.status = "confirmed"
        elif self.status in ("confirmed", "coasting"):
            self.status = "confirmed" if accepted else "coasting"

    def _should_drop(self):
        cfg = self.cfg
        if self.since_update_s > cfg.max_coast_s or self.kf.sigma_gap > cfg.max_gap_sigma_m:
            return True
        # A tentative track that can no longer reach M hits in its window is noise.
        if self.status == "tentative" and len(self.history) == self.history.maxlen:
            return sum(self.history) < cfg.confirm_hits
        return False

    def _estimate(self, updated_by, rejected, nis):
        if self.kf is None:
            return TrackEstimate("none", rejected=rejected)
        return TrackEstimate(
            status=self.status,
            gap_m=self.kf.gap,
            gap_rate_mps=self.kf.gap_rate,
            sigma_gap_m=self.kf.sigma_gap,
            sigma_rate_mps=self.kf.sigma_rate,
            updated_by=updated_by,
            rejected=rejected,
            nis=nis,
        )


class RawEstimator:
    """Baseline without a filter: the latest measurement as is.

    camera_raw infers the closing speed by differencing two consecutive camera gaps, which
    is what a naive camera-only system would do (and why its TTC is so noisy).
    """

    def __init__(self, source):
        if source not in ("camera", "radar"):
            raise ValueError(source)
        self.source = source
        self._previous = None   # (gap, seconds since)

    def reset(self):
        self._previous = None

    def step(self, dt, camera=None, radar=None):
        m = camera if self.source == "camera" else radar
        if m is None:
            if self._previous is not None:
                gap, age = self._previous
                self._previous = (gap, age + dt) if age + dt < 0.2 else None
            return TrackEstimate("none")

        rate = m.rate_mps
        if self.source == "camera":
            rate = None
            if self._previous is not None:
                prev_gap, age = self._previous
                rate = (m.gap_m - prev_gap) / (age + dt)
            self._previous = (m.gap_m, 0.0)
        return TrackEstimate("raw", gap_m=m.gap_m, gap_rate_mps=rate, sigma_gap_m=m.sigma_gap_m,
                             sigma_rate_mps=m.sigma_rate_mps, updated_by=(m.source,))


def make_estimator(mode, cfg):
    if mode == "camera_raw":
        return RawEstimator("camera")
    if mode == "radar_raw":
        return RawEstimator("radar")
    return LeadTracker(cfg, mode)
