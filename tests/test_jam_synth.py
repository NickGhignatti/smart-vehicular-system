"""Jam scenarios in the synthetic world: the column is rendered and its waves reach the ego."""

import numpy as np
import pytest

from svs.config import load_config
from svs.scenarios import PHANTOM_CRUISE_MPS, make_scenario
from svs.synth import Vehicle, _render_radar, generate

CFG = load_config()
DT = CFG.carla.fixed_delta_seconds


def first_stop(speeds):
    """Frame where the speed first falls below 0.5 m/s after having been above 2 m/s."""
    speeds = np.nan_to_num(np.asarray(speeds, dtype=float))
    moving = int(np.argmax(speeds > 2.0))
    return moving + int(np.argmax(speeds[moving:] < 0.5))


def test_queue_stop_wave_reaches_the_car_ahead_of_the_ego_late_and_nobody_collides():
    plan = make_scenario("jam_queue", 1, 40.0, DT)
    rec = generate("jam_queue", 1, CFG, 40.0)
    lead_speed = [f.truth.lead_speed_mps if f.truth else np.nan for f in rec.frames]
    assert (first_stop(lead_speed) - first_stop(plan.lead_speed)) * DT >= 1.0       # five IDM drivers sit in between
    assert min(f.truth.gap_m for f in rec.frames if f.truth) > 0.3


def test_phantom_brake_reaches_the_car_in_front_of_the_ego():
    seconds = 40.0
    event = make_scenario("phantom_jam", 3, seconds, DT).events_of("phantom_brake")[0]
    rec = generate("phantom_jam", 3, CFG, seconds)
    assert rec.frames[0].truth.ego_speed_mps > 0.9 * PHANTOM_CRUISE_MPS             # starts in motion
    after = [f.truth.lead_speed_mps for f in rec.frames[int(event.t_start / DT):] if f.truth]
    assert min(after) < 0.9 * PHANTOM_CRUISE_MPS


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_phantom_jam_puts_the_ego_in_a_critical_situation(seed):
    rec = generate("phantom_jam", seed, CFG, 60.0)
    ttc = np.array([f.truth.ttc_s if f.truth else np.inf for f in rec.frames])
    assert (ttc < CFG.warning.label_ttc_s).sum() * DT >= 0.4        # otherwise the scenario cannot test any warning
    assert min(f.truth.gap_m for f in rec.frames if f.truth) > 0.0  # ...but it must stay a near miss, not an overlap


def test_neighbors_in_the_adjacent_lane_are_seen_by_the_camera():
    rec = generate("jam_neighbors", 1, CFG, 5.0)
    assert len(rec.frames[0].detections) >= 2


def test_cut_in_in_a_jam_puts_a_nearer_car_in_front_of_the_ego():
    seconds = 46.0
    event = make_scenario("jam_cut_in", 3, seconds, DT).events_of("cut_in")[0]
    rec = generate("jam_cut_in", 3, CFG, seconds)
    gaps = np.array([f.truth.gap_m if f.truth else np.nan for f in rec.frames])
    before, after = int(event.t_start / DT) - 1, int(event.t_end / DT) + 10
    assert gaps[after] < gaps[before]
    assert np.nanmin(gaps) > 0.3


def test_frames_are_numbered_consecutively_in_a_column():
    rec = generate("jam_queue", 1, CFG, 5.0)
    assert [f.frame for f in rec.frames] == list(range(len(rec.frames)))


def test_radar_does_not_see_through_the_car_ahead():
    rng = np.random.default_rng(0)

    def fast_points(lateral):
        """Points of a far car moving away fast (clutter and the near car move far slower)."""
        cars = [Vehicle(10.0, 0.0, 2.0, "lead"), Vehicle(25.0, lateral, 12.0, "column")]
        pts = np.vstack([_render_radar(CFG, cars, 0.0, 3.0, 0.0, rng) for _ in range(20)])
        return int((pts[:, 3] > 5.0).sum())

    assert fast_points(0.0) == 0           # right behind the near car: hidden
    assert fast_points(3.0) > 0            # far enough to the side to be hit by the rays
