"""Unit tests for svs.sim.ground_truth, on fake actors (no simulator needed).

Run from the project root:  python -m unittest discover -s tests -v
"""

import unittest
from types import SimpleNamespace

from svs.sim.ground_truth import lead_truth


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)


def make_actor(location, forward, right, half_length_m=2.0):
    transform = SimpleNamespace(
        location=Vector(*location),
        get_forward_vector=lambda: Vector(*forward),
        get_right_vector=lambda: Vector(*right),
    )
    return SimpleNamespace(
        get_transform=lambda: transform,
        get_location=lambda: transform.location,
        get_velocity=lambda: Vector(),
        bounding_box=SimpleNamespace(extent=SimpleNamespace(x=half_length_m)),
    )


class LeadTruthLateralTest(unittest.TestCase):
    def test_lateral_is_positive_to_the_right_and_negative_to_the_left(self):
        ego = make_actor((0, 0, 0), forward=(1, 0, 0), right=(0, 1, 0))
        on_right = make_actor((20, 1.5, 0), forward=(1, 0, 0), right=(0, 1, 0))
        on_left = make_actor((20, -1.5, 0), forward=(1, 0, 0), right=(0, 1, 0))
        self.assertAlmostEqual(lead_truth(ego, on_right).lateral_m, 1.5)
        self.assertAlmostEqual(lead_truth(ego, on_left).lateral_m, -1.5)

    def test_lateral_follows_the_ego_heading(self):
        # ego yawed 90 deg: forward is +y, right is -x (CARLA is left-handed)
        ego = make_actor((0, 0, 0), forward=(0, 1, 0), right=(-1, 0, 0))
        lead = make_actor((-1.5, 20, 0), forward=(0, 1, 0), right=(-1, 0, 0))
        truth = lead_truth(ego, lead)
        self.assertAlmostEqual(truth.lateral_m, 1.5)
        self.assertAlmostEqual(truth.gap_m, 16.0)


if __name__ == "__main__":
    unittest.main()
