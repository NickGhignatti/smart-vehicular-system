"""Scenario definitions shared by the synthetic world and the CARLA runner (no CARLA here).

A scenario fixes, for a given seed:
- the lead speed profile over time (stop-and-go cycles, a parked lead, ...);
- the ego driver: an IDM car-following driver with scripted "distraction" windows during
  which it does not react. Those windows create the genuinely critical situations that a
  warning *should* flag; outside them the driver keeps a safe distance and any warning is
  a false warning;
- scripted events: camera occlusion, an adjacent car overtaking, a cut-in.

Scenarios:
    approach     parked lead, ego drives towards it and stops (sensor calibration)
    stop_and_go  lead cycles 0 -> ~30 km/h -> 0, ego follows (main study)
    occlusion    stop_and_go + camera occlusions and an adjacent car passing by
    cut_in       stop_and_go + a car cutting in between ego and lead
"""

import math
from dataclasses import dataclass, field

import numpy as np

SCENARIOS = ("approach", "stop_and_go", "occlusion", "cut_in")


@dataclass(frozen=True)
class Event:
    t_start: float
    t_end: float
    kind: str          # "distraction", "camera_occlusion", "partial_occlusion", "overtake", "cut_in"
    value: float = 0.0  # kind-specific (e.g. occluded fraction)

    def active(self, t):
        return self.t_start <= t < self.t_end

    def as_list(self):
        return [self.t_start, self.t_end, self.kind, self.value]


@dataclass
class ScenarioPlan:
    name: str
    seed: int
    duration_s: float
    dt: float
    lead_speed: np.ndarray            # target lead speed per frame, m/s
    initial_gap_m: float
    ego_initial_speed_mps: float
    events: list = field(default_factory=list)

    def events_of(self, kind):
        return [e for e in self.events if e.kind == kind]

    def event_active(self, kind, t):
        return any(e.active(t) for e in self.events_of(kind))


def _stop_and_go_profile(n, dt, rng, v_high=8.3):
    """Piecewise lead speed: accelerate, cruise, brake to a stop, wait; repeated with jitter."""
    speed = np.zeros(n)
    t, v, i = 0.0, 0.0, 0
    phase = "wait"
    phase_left = rng.uniform(1.0, 3.0)
    target = v_high
    accel = 1.5
    while i < n:
        if phase_left <= 0:
            if phase == "wait":
                phase, target, accel = "accelerate", rng.uniform(0.7, 1.0) * v_high, rng.uniform(1.0, 2.0)
                phase_left = math.inf
            elif phase == "accelerate":
                phase, phase_left = "cruise", rng.uniform(3.0, 8.0)
            elif phase == "cruise":
                phase, accel, phase_left = "brake", rng.uniform(1.5, 3.5), math.inf
            elif phase == "brake":
                phase, phase_left = "wait", rng.uniform(2.0, 5.0)
        if phase == "accelerate":
            v = min(target, v + accel * dt)
            if v >= target:
                phase_left = 0
        elif phase == "brake":
            v = max(0.0, v - accel * dt)
            if v <= 0.0:
                phase_left = 0
        else:
            phase_left -= dt
        speed[i] = v
        i += 1
        t += dt
    return speed


def _distractions(duration, rng, first=8.0, every=(14.0, 22.0), length=(2.0, 3.5)):
    events, t = [], first + rng.uniform(0.0, 4.0)
    while t < duration - 5.0:
        d = rng.uniform(*length)
        events.append(Event(t, t + d, "distraction"))
        t += d + rng.uniform(*every)
    return events


def make_scenario(name, seed, duration_s=90.0, dt=0.05):
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}; choose from {SCENARIOS}")
    rng = np.random.default_rng(seed)
    n = int(round(duration_s / dt))

    if name == "approach":
        return ScenarioPlan(name, seed, duration_s, dt, np.zeros(n), initial_gap_m=50.0,
                            ego_initial_speed_mps=0.0)

    plan = ScenarioPlan(name, seed, duration_s, dt, _stop_and_go_profile(n, dt, rng),
                        initial_gap_m=float(rng.uniform(10.0, 15.0)), ego_initial_speed_mps=0.0,
                        events=_distractions(duration_s, rng))
    if name == "occlusion":
        t = 6.0
        while t < duration_s - 6.0:
            t += rng.uniform(8.0, 14.0)
            kind = "camera_occlusion" if rng.random() < 0.5 else "partial_occlusion"
            length = rng.uniform(0.8, 2.0)
            plan.events.append(Event(t, t + length, kind, value=float(rng.uniform(0.3, 0.6))))
        plan.events.append(Event(15.0, 25.0, "overtake"))
    elif name == "cut_in":
        t0 = float(rng.uniform(25.0, 40.0))
        plan.events.append(Event(t0, t0 + 3.0, "cut_in"))
    plan.events.sort(key=lambda e: e.t_start)
    return plan


# --- ego driver -------------------------------------------------------------------------

@dataclass(frozen=True)
class DriverParams:
    """Intelligent Driver Model (Treiber 2000), a standard human car-following model."""
    desired_speed: float = 30 / 3.6
    time_gap: float = 1.6
    min_gap: float = 4.0
    max_accel: float = 1.2
    comfort_decel: float = 2.0
    max_decel: float = 7.0
    reaction_s: float = 0.4
    # A distracted driver still notices when the lead is about to be hit (late reaction).
    wake_up_ttc_s: float = 1.2


class DriverModel:
    """IDM driver; during "distraction" events it keeps its speed until the TTC gets very small."""

    def __init__(self, plan, params=None):
        self.plan = plan
        self.p = params or DriverParams()
        self._delay = max(1, int(round(self.p.reaction_s / plan.dt)))
        self._seen = []

    def accel(self, t, ego_speed, gap, lead_speed):
        """Desired acceleration given what the driver perceives (gap = None: no lead)."""
        p = self.p
        # Perception with reaction delay.
        self._seen.append((gap, lead_speed))
        gap, lead_speed = self._seen[-self._delay] if len(self._seen) >= self._delay else self._seen[0]

        free = p.max_accel * (1.0 - (ego_speed / p.desired_speed) ** 4)
        if gap is None:
            return free
        if self.plan.event_active("distraction", t):
            closing = ego_speed - lead_speed
            if closing <= 0.1 or gap / closing > p.wake_up_ttc_s:
                return min(free, 0.0) if ego_speed > 0.5 else free * 0.3
        dv = ego_speed - lead_speed
        s_star = p.min_gap + max(0.0, ego_speed * p.time_gap + ego_speed * dv / (2 * math.sqrt(p.max_accel * p.comfort_decel)))
        a = p.max_accel * (1.0 - (ego_speed / p.desired_speed) ** 4 - (s_star / max(gap, 0.5)) ** 2)
        return float(np.clip(a, -p.max_decel, p.max_accel))


def approach_accel(gap_m, ego_speed_mps, cruise_mps=15 / 3.6, stop_gap_m=5.0, max_decel=2.5):
    """Calibration driver: cruise slowly towards the parked lead and stop smoothly at stop_gap_m."""
    if gap_m is None:
        return 0.5
    remaining = gap_m - stop_gap_m
    if remaining <= 0.2:
        return -max_decel if ego_speed_mps > 0.05 else 0.0
    needed = ego_speed_mps ** 2 / (2.0 * remaining)        # constant decel that stops on the mark
    if needed > 0.8:
        return -min(needed, max_decel)
    return float(np.clip(0.5 * (cruise_mps - ego_speed_mps), -1.0, 1.0))
