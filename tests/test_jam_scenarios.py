"""Jam scenarios: a column of IDM cars ahead of the ego (no CARLA needed)."""

import dataclasses

import numpy as np
import pytest

from svs.scenarios import (JAM_CARS, PHANTOM_CRUISE_MPS, SCENARIOS, DriverModel, follower_driver,
                           follower_params, make_scenario)

JAMS = ("jam_queue", "jam_neighbors", "jam_cut_in", "phantom_jam")
CAR_LENGTH_M = 4.5


def simulate_column(plan):
    """Car 0 follows the plan's lead profile, car i follows car i-1 with its own IDM driver.

    Returns (speeds, gaps), both (frames, cars); gaps[:, 0] is unused (nan).
    """
    n, dt = plan.n_ahead, plan.dt
    drivers = [None] + [follower_driver(plan, i) for i in range(1, n)]
    x = [-(plan.initial_gap_m + CAR_LENGTH_M) * i for i in range(n)]
    v = [plan.ego_initial_speed_mps] * n
    speeds, gaps = np.zeros((len(plan.lead_speed), n)), np.full((len(plan.lead_speed), n), np.nan)
    for k, target in enumerate(plan.lead_speed):
        a = [0.0] + [drivers[i].accel(k * dt, v[i], x[i - 1] - x[i] - CAR_LENGTH_M, v[i - 1]) for i in range(1, n)]
        v[0] = float(target)
        for i in range(1, n):
            v[i] = max(0.0, v[i] + a[i] * dt)
        for i in range(n):
            x[i] += v[i] * dt
        speeds[k] = v
        gaps[k, 1:] = [x[i - 1] - x[i] - CAR_LENGTH_M for i in range(1, n)]
    return speeds, gaps


def test_jam_scenarios_describe_a_column():
    for name in JAMS:
        assert name in SCENARIOS
        a, b = make_scenario(name, 5), make_scenario(name, 5)
        assert a.n_ahead == JAM_CARS and a.events == b.events and np.array_equal(a.lead_speed, b.lead_speed)
    assert make_scenario("jam_neighbors", 5).neighbor_cars > 0
    assert make_scenario("jam_cut_in", 5).events_of("cut_in")
    assert 4.0 <= make_scenario("jam_queue", 5).initial_gap_m <= 7.0       # bumper to bumper, much tighter than stop_and_go
    assert make_scenario("jam_cut_in", 5).initial_gap_m >= 9.0             # a car must fit between the ego and the column


def test_single_lead_scenarios_are_unchanged():
    for name in ("approach", "stop_and_go", "occlusion", "cut_in"):
        plan = make_scenario(name, 5)
        assert plan.n_ahead == 1 and plan.neighbor_cars == 0 and not plan.events_of("phantom_brake")
    assert not make_scenario("jam_queue", 5).events_of("phantom_brake")


def test_follower_params_are_seeded_and_differ_per_car():
    plan = make_scenario("jam_queue", 3)
    assert follower_params(plan, 2) == follower_params(plan, 2)
    assert follower_params(plan, 1) != follower_params(plan, 2)
    for i in range(1, plan.n_ahead):
        p = follower_params(plan, i)
        assert 1.0 <= p.time_gap <= 1.6 and 2.0 <= p.min_gap <= 3.0 and 0.3 <= p.reaction_s <= 0.7


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_queue_runs_a_full_stop_and_go_cycle_down_the_column_without_collisions(seed):
    speeds, gaps = simulate_column(make_scenario("jam_queue", seed))
    assert np.nanmin(gaps) > 0.5
    last = speeds[:, -1]
    first_moving = int(np.argmax(last > 2.0))
    assert last.max() > 2.0 and last[first_moving:].min() < 0.2           # the back car moves, then stops again


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_phantom_brake_travels_backwards_only(seed):
    plan = make_scenario("phantom_jam", seed)
    event = plan.events_of("phantom_brake")[0]
    braking, start = int(event.value), int(event.t_start / plan.dt)
    assert 1 <= braking < plan.n_ahead - 1
    speeds, gaps = simulate_column(plan)
    assert np.nanmin(gaps) > 0.5
    assert np.allclose(speeds[:, 0], PHANTOM_CRUISE_MPS)
    assert speeds[start:, :braking].min() > 0.95 * PHANTOM_CRUISE_MPS     # after the event, cars ahead of the culprit don't slow down
    assert speeds[:, braking].min() < 0.5 * PHANTOM_CRUISE_MPS
    assert speeds[:, braking + 1].min() < 0.8 * PHANTOM_CRUISE_MPS        # the car behind has to react


def test_ego_driver_ignores_the_phantom_event():
    plan = make_scenario("phantom_jam", 1)
    e = plan.events_of("phantom_brake")[0]
    ego = DriverModel(dataclasses.replace(plan, events=[e]))               # car_index None: not part of the column
    assert ego.accel(e.t_start + 0.1, PHANTOM_CRUISE_MPS, 12.0, PHANTOM_CRUISE_MPS) > -1.0
