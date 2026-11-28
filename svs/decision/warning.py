"""Time-to-collision and a stable forward-collision warning.

TTC (Lab5): gap / closing speed while closing in, infinite otherwise. Time headway
(gap / ego speed) is reported too: in stop-and-go traffic the gap is often constant
(TTC = inf) while the headway still says whether we follow too close.

Stability, the difference with Lab5's plain threshold:
- hysteresis: raise below ttc_on, clear only above ttc_off (> ttc_on);
- persistence: TTC must stay below ttc_on for on_frames consecutive frames;
- minimum duration: once raised, keep it at least min_on_s (no blinking);
- only on a confirmed track (svs.fusion.tracker), so one noisy frame cannot warn.
"""

import math
from dataclasses import dataclass


def time_to_collision(gap_m, closing_speed_mps, min_closing_mps=0.5):
    if gap_m is None or closing_speed_mps is None or closing_speed_mps <= min_closing_mps:
        return math.inf
    return max(gap_m, 0.0) / closing_speed_mps


def time_headway(gap_m, ego_speed_mps, min_speed_mps=0.5):
    if gap_m is None or ego_speed_mps <= min_speed_mps:
        return math.inf
    return max(gap_m, 0.0) / ego_speed_mps


@dataclass(frozen=True)
class WarningState:
    active: bool
    ttc_s: float
    headway_s: float
    raised: bool    # rising edge this frame
    cleared: bool   # falling edge this frame


class ForwardCollisionWarning:
    def __init__(self, cfg):
        self.cfg = cfg
        self.active = False
        self._below = 0
        self._on_for_s = 0.0

    def reset(self):
        self.active = False
        self._below = 0
        self._on_for_s = 0.0

    def step(self, dt, estimate, ego_speed_mps):
        """estimate: TrackEstimate; returns the WarningState of this frame."""
        cfg = self.cfg
        if estimate is not None and estimate.confirmed:
            ttc = time_to_collision(estimate.gap_m, estimate.closing_speed_mps, cfg.min_closing_mps)
            headway = time_headway(estimate.gap_m, ego_speed_mps)
        else:
            ttc = headway = math.inf

        was_active = self.active
        self._below = self._below + 1 if ttc < cfg.ttc_on_s else 0
        if not self.active:
            if self._below >= cfg.on_frames:
                self.active = True
                self._on_for_s = 0.0
        else:
            self._on_for_s += dt
            if ttc > cfg.ttc_off_s and self._on_for_s >= cfg.min_on_s:
                self.active = False

        return WarningState(self.active, ttc, headway,
                            raised=self.active and not was_active,
                            cleared=was_active and not self.active)
