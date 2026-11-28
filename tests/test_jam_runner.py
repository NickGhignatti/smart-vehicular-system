"""Jam scenarios in the fake CARLA: spawning the column and driving it from run_scenario."""

import dataclasses
import random

import carla
import pytest

from svs.apps import run_scenario
from svs.config import load_config
from svs.scenarios import JAM_CARS, PHANTOM_CRUISE_MPS, make_scenario
from svs.sim.spawn import spawn_column
from tests import fake_carla

FAKE_CAR_LENGTH_M = 4.8   # 2 * the half-length of the fake bounding box


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    monkeypatch.setattr(carla, "Client", fake_carla.FakeClient)
    fake_carla.FakeClient.world = None
    return dataclasses.replace(load_config(), output_dir=tmp_path, sensor_timeout=0.5)


@pytest.fixture
def frames(monkeypatch):
    """Every tick of the fake world: [(x, y, speed)] of all vehicles, front to back."""
    seen, tick = [], fake_carla.FakeWorld.tick

    def spy(self, *args, **kwargs):
        cars = [a for a in self.actors if isinstance(a, fake_carla.FakeVehicle)]
        seen.append([(c.x, c.y, c.v) for c in sorted(cars, key=lambda c: -c.x)])
        return tick(self, *args, **kwargs)

    monkeypatch.setattr(fake_carla.FakeWorld, "tick", spy)
    return seen


def run(cfg, scenario, seed, seconds):
    return run_scenario.run(cfg, scenario=scenario, seed=seed, seconds=seconds, mode="fused",
                            control="warn", display=False)


def lane_gaps(frame, lane_y=0.0):
    """Bumper gaps between consecutive cars of one lane."""
    cars = [c for c in frame if abs(c[1] - lane_y) < 1.0]
    return [a[0] - b[0] - FAKE_CAR_LENGTH_M for a, b in zip(cars, cars[1:])]


def test_spawn_column_places_the_cars_front_to_back_at_speed():
    world = fake_carla.FakeWorld()
    ego, column, _ = spawn_column(world, random.Random(1), n_ahead=4, gap_m=9.5, straight_len_m=150.0, speed_mps=6.0)
    xs = [c.x for c in column]
    assert xs == sorted(xs, reverse=True)                                  # column[0] is the leader, column[-1] is next to the ego
    assert all(abs(a - b - 9.5) < 1e-6 for a, b in zip(xs, [*xs[1:], ego.x]))
    assert all(car.v == 6.0 for car in (ego, *column))


def test_spawn_column_cleans_up_when_the_road_is_too_short():
    world = fake_carla.FakeWorld()
    with pytest.raises(RuntimeError):
        spawn_column(world, random.Random(1), n_ahead=300, gap_m=9.5, straight_len_m=150.0)
    assert world.actors == []


def test_jam_queue_runs_a_column_without_collisions(cfg, frames):
    run(cfg, "jam_queue", 1, 20.0)
    assert max(len(f) for f in frames) == 1 + JAM_CARS
    assert min(min(lane_gaps(f)) for f in frames) > 0.3


def test_jam_neighbors_adds_cars_in_the_adjacent_lane(cfg, frames):
    run(cfg, "jam_neighbors", 1, 10.0)
    assert max(len(f) for f in frames) == 1 + JAM_CARS + make_scenario("jam_neighbors", 1).neighbor_cars
    assert sum(abs(c[1] - 3.5) < 1.0 for c in frames[-1]) == 3


def test_phantom_jam_starts_at_cruise_and_the_brake_hits_the_planned_car(cfg, frames):
    seconds = 40.0                                  # the event times depend on the duration: plan and run must agree
    plan = make_scenario("phantom_jam", 3, seconds)
    event = plan.events_of("phantom_brake")[0]
    run(cfg, "phantom_jam", 3, seconds)
    assert frames[0][-1][2] > 0.9 * PHANTOM_CRUISE_MPS                       # the ego does not start from standstill
    after = frames[int(event.t_start / plan.dt):]
    assert min(f[int(event.value)][2] for f in after) < 0.7 * PHANTOM_CRUISE_MPS
    assert min(f[0][2] for f in after) > 0.9 * PHANTOM_CRUISE_MPS            # the leader never slows


def test_jam_cut_in_squeezes_a_car_in_front_of_the_ego(cfg, frames):
    run(cfg, "jam_cut_in", 3, 46.0)                 # the cut-in starts between 25 s and 40 s
    assert max(len(f) for f in frames) == 1 + JAM_CARS + 1
