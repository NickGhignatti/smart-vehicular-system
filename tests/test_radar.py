"""Unit tests for svs.perception.radar, on synthetic detections (no simulator needed).

Run from the project root:  python -m unittest discover -s tests -v
(importing svs.perception.radar still imports `carla` through svs.sim.sensors, so the
package must be installed, but no server is required).
"""

import unittest

import numpy as np

from svs.perception.radar import (RadarMeasurement, _front_corridor_mask, _to_cartesian,
                                  extract_radar_measurements)


def make_points(*rows):
    """Build an (N, 4) array [depth, azimuth, altitude, velocity] from (depth, az_deg, alt_deg, vel)."""
    return np.array(
        [(depth, np.radians(az), np.radians(alt), vel) for depth, az, alt, vel in rows],
        dtype=np.float64,
    ).reshape(-1, 4)


class ToCartesianTest(unittest.TestCase):
    def test_straight_ahead_point(self):
        xyz = _to_cartesian(make_points((20.0, 0.0, 0.0, 0.0)))
        np.testing.assert_allclose(xyz, [[20.0, 0.0, 0.0]], atol=1e-9)

    def test_azimuth_moves_point_sideways(self):
        xyz = _to_cartesian(make_points((30.0, 5.0, 0.0, 0.0)))
        np.testing.assert_allclose(xyz, [[29.8858, 2.6147, 0.0]], atol=1e-3)

    def test_altitude_moves_point_up(self):
        xyz = _to_cartesian(make_points((10.0, 0.0, 10.0, 0.0)))
        np.testing.assert_allclose(xyz, [[9.8481, 0.0, 1.7365]], atol=1e-3)

    def test_preserves_distance_from_sensor(self):
        points = make_points((12.0, 7.0, -3.0, 0.0), (45.0, -14.0, 4.0, 0.0), (3.0, 1.0, 0.5, 0.0))
        xyz = _to_cartesian(points)
        np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), points[:, 0], atol=1e-9)

    def test_shape_and_row_order(self):
        points = make_points((10.0, 0.0, 0.0, 0.0), (20.0, 0.0, 0.0, 0.0), (30.0, 0.0, 0.0, 0.0))
        xyz = _to_cartesian(points)
        self.assertEqual(xyz.shape, (3, 3))
        np.testing.assert_allclose(xyz[:, 0], [10.0, 20.0, 30.0], atol=1e-9)

    def test_does_not_modify_input(self):
        points = make_points((12.0, 7.0, -3.0, 1.5))
        before = points.copy()
        _to_cartesian(points)
        np.testing.assert_array_equal(points, before)


class FrontCorridorMaskTest(unittest.TestCase):
    def test_real_life_scene(self):
        points = make_points(
            (23.0, 1.0, -1.0, 0.0),    # lead rear bumper          -> keep
            (23.2, -2.0, 1.5, 0.0),    # lead rear window          -> keep
            (18.0, 12.0, 0.0, 0.0),    # car in the next lane      -> drop (|y| too large)
            (12.7, 0.0, -4.5, 0.0),    # road surface              -> drop (z too low)
            (25.0, 14.0, 0.0, 0.0),    # guardrail                 -> drop (|y| too large)
            (0.6, 0.0, 0.0, 0.0),      # reflection on own hood    -> drop (too close)
        )
        mask = _front_corridor_mask(_to_cartesian(points))
        self.assertEqual(mask.tolist(), [True, True, False, False, False, False])

    def test_returns_one_boolean_per_point(self):
        mask = _front_corridor_mask(_to_cartesian(make_points((10, 0, 0, 0), (20, 0, 0, 0))))
        self.assertEqual(mask.shape, (2,))
        self.assertEqual(mask.dtype, bool)

    def test_lateral_limit_is_symmetric(self):
        # y = 20 * sin(az): +-3 deg -> +-1.05 m (inside), +-6 deg -> +-2.09 m (outside).
        points = make_points((20, 3, 0, 0), (20, -3, 0, 0), (20, 6, 0, 0), (20, -6, 0, 0))
        mask = _front_corridor_mask(_to_cartesian(points))
        self.assertEqual(mask.tolist(), [True, True, False, False])

    def test_corridor_has_constant_width_with_distance(self):
        # Same lateral offset (1.0 m) at 5 m and 40 m: both must be kept, unlike a fixed angle cone.
        xyz = np.array([[5.0, 1.0, 0.0], [40.0, 1.0, 0.0]])
        self.assertEqual(_front_corridor_mask(xyz).tolist(), [True, True])

    def test_custom_limits(self):
        xyz = np.array([[10.0, 1.0, 0.0]])
        self.assertTrue(_front_corridor_mask(xyz, half_width_m=1.5)[0])
        self.assertFalse(_front_corridor_mask(xyz, half_width_m=0.5)[0])

    def test_empty_input(self):
        mask = _front_corridor_mask(np.empty((0, 3)))
        self.assertEqual(mask.shape, (0,))


class ExtractRadarMeasurementsTest(unittest.TestCase):
    def lead_points(self, velocity=4.2):
        """Points on a lead 23-25 m ahead, straight in front of the sensor."""
        return [(23.0, 0.0, 0.0, velocity), (23.1, 0.0, 0.0, velocity), (23.4, 0.0, 0.0, velocity),
                (24.0, 0.0, 0.0, velocity), (25.0, 0.0, 0.0, velocity)]

    def test_empty_array_returns_none(self):
        self.assertIsNone(extract_radar_measurements(np.empty((0, 4))))

    def test_no_point_in_corridor_returns_none(self):
        points = make_points((18.0, 12.0, 0.0, 1.0), (25.0, 14.0, 0.0, 1.0), (0.5, 0.0, 0.0, 1.0))
        self.assertIsNone(extract_radar_measurements(points))

    def test_returns_radar_measurement(self):
        result = extract_radar_measurements(make_points(*self.lead_points()))
        self.assertIsInstance(result, RadarMeasurement)

    def test_percentile_zero_gives_nearest_point(self):
        result = extract_radar_measurements(make_points(*self.lead_points()), percentile=0)
        self.assertAlmostEqual(result.range_m, 23.0, places=6)

    def test_percentile_fifty_gives_median(self):
        result = extract_radar_measurements(make_points(*self.lead_points()), percentile=50)
        self.assertAlmostEqual(result.range_m, 23.4, places=6)

    def test_default_percentile_is_between_nearest_point_and_median(self):
        result = extract_radar_measurements(make_points(*self.lead_points()))
        self.assertGreaterEqual(result.range_m, 23.0)
        self.assertLessEqual(result.range_m, 23.4)

    def test_range_is_longitudinal_not_ray_length(self):
        # depth 10 m at 5 deg (y = 0.87 m, inside the corridor): forward distance is 10 * cos(5 deg).
        points = make_points((10.0, 5.0, 0.0, 0.0))
        result = extract_radar_measurements(points, percentile=50)
        self.assertAlmostEqual(result.range_m, 10.0 * np.cos(np.radians(5.0)), places=6)

    def test_range_rate_sign_is_flipped(self):
        # CARLA velocity is positive towards the sensor; range_rate is negative when closing in.
        closing = extract_radar_measurements(make_points(*self.lead_points(velocity=4.2)))
        opening = extract_radar_measurements(make_points(*self.lead_points(velocity=-2.0)))
        self.assertAlmostEqual(closing.range_rate_mps, -4.2, places=6)
        self.assertAlmostEqual(opening.range_rate_mps, 2.0, places=6)

    def test_range_rate_zero_when_same_speed(self):
        result = extract_radar_measurements(make_points(*self.lead_points(velocity=0.0)))
        self.assertAlmostEqual(result.range_rate_mps, 0.0, places=9)

    def test_range_rate_corrects_projection_on_axis(self):
        # Lead closing in at 4.2 m/s along the road, seen at different angles inside the corridor:
        # the radial velocity is 4.2 * cos(angle), the corrected longitudinal one must be 4.2.
        rows = [(25.0, az, 0.0, 4.2 * np.cos(np.radians(az))) for az in (-3.0, 0.0, 2.0, 3.0)]
        result = extract_radar_measurements(make_points(*rows))
        self.assertAlmostEqual(result.range_rate_mps, -4.2, places=6)

    def test_range_rate_is_robust_to_a_spurious_velocity(self):
        rows = self.lead_points(velocity=4.2) + [(24.0, 0.0, 0.0, 40.0)]
        result = extract_radar_measurements(make_points(*rows))
        self.assertAlmostEqual(result.range_rate_mps, -4.2, places=6)

    def test_points_outside_corridor_do_not_affect_measurement(self):
        clean = extract_radar_measurements(make_points(*self.lead_points()))
        clutter = [
            (18.0, 12.0, 0.0, 99.0),   # next lane
            (12.7, 0.0, -4.5, 99.0),   # road
            (25.0, 14.0, 0.0, 99.0),   # guardrail
            (0.6, 0.0, 0.0, 99.0),     # own hood
        ]
        noisy = extract_radar_measurements(make_points(*(self.lead_points() + clutter)))
        self.assertEqual(noisy, clean)

    def test_n_points_counts_only_corridor_detections(self):
        rows = self.lead_points() + [(18.0, 12.0, 0.0, 1.0), (0.6, 0.0, 0.0, 1.0)]
        result = extract_radar_measurements(make_points(*rows))
        self.assertEqual(result.n_points, 5)

    def test_lateral_is_median_and_keeps_its_sign(self):
        # y = 20 * sin(3 deg) = 1.047 m, all on the same side.
        rows = [(20.0, 3.0, 0.0, 0.0)] * 3
        result = extract_radar_measurements(make_points(*rows))
        self.assertAlmostEqual(result.lateral_m, 20.0 * np.sin(np.radians(3.0)), places=6)

        mirrored = extract_radar_measurements(make_points(*[(20.0, -3.0, 0.0, 0.0)] * 3))
        self.assertAlmostEqual(mirrored.lateral_m, -result.lateral_m, places=9)

    def test_values_are_plain_python_numbers(self):
        result = extract_radar_measurements(make_points(*self.lead_points()))
        self.assertIs(type(result.range_m), float)
        self.assertIs(type(result.range_rate_mps), float)
        self.assertIs(type(result.lateral_m), float)
        self.assertIs(type(result.n_points), int)

    def test_does_not_modify_input(self):
        points = make_points(*self.lead_points())
        before = points.copy()
        extract_radar_measurements(points)
        np.testing.assert_array_equal(points, before)


if __name__ == "__main__":
    unittest.main()
