import math

import numpy as np
import pytest

from svs.perception.camera_geometry import PinholeCamera, focal_length_px
from tests.synthetic import CAMERA, car_box


def test_focal_length_from_fov():
    assert focal_length_px(640, 90.0) == pytest.approx(320.0)
    assert focal_length_px(640, 60.0) == pytest.approx(554.26, abs=0.01)


def test_point_on_optical_axis_projects_to_image_center():
    cam = PinholeCamera(640, 360, 60.0, 1.4, pitch_deg=0.0)
    u, v = cam.project([[10.0, 0.0, 0.0]])[0]
    assert (u, v) == pytest.approx((cam.cx, cam.cy))


def test_point_behind_camera_is_nan():
    assert np.isnan(CAMERA.project([[-5.0, 0.0, 0.0]])).all()


@pytest.mark.parametrize("pitch", [0.0, -6.0, 3.0])
@pytest.mark.parametrize("distance", [3.0, 10.0, 25.0, 50.0])
def test_ground_distance_inverts_projection(pitch, distance):
    cam = PinholeCamera(640, 360, 60.0, 1.4, pitch)
    v = cam.project([[distance, 0.0, -cam.mount_height_m]])[0, 1]
    assert cam.ground_distance(v) == pytest.approx(distance, rel=1e-9)


def test_ground_distance_above_horizon_is_none():
    assert CAMERA.ground_distance(0.0) is None


def test_ground_sigma_matches_numerical_derivative():
    v = CAMERA.project([[20.0, 0.0, -1.4]])[0, 1]
    h = 1e-3
    numeric = abs(CAMERA.ground_distance(v + h) - CAMERA.ground_distance(v - h)) / (2 * h)
    assert CAMERA.ground_distance_sigma(v, 1.0) == pytest.approx(numeric, rel=1e-4)


def test_ground_sigma_grows_with_distance_squared():
    def sigma_at(d):
        return CAMERA.ground_distance_sigma(CAMERA.project([[d, 0.0, -1.4]])[0, 1], 1.0)

    assert sigma_at(20.0) / sigma_at(10.0) == pytest.approx(4.0, rel=0.1)


def test_min_ground_distance_is_bottom_row():
    d = CAMERA.min_ground_distance()
    v = CAMERA.project([[d, 0.0, -1.4]])[0, 1]
    assert v == pytest.approx(CAMERA.height - 0.5)


@pytest.mark.parametrize("distance", [5.0, 15.0, 40.0])
def test_width_distance_recovers_distance(distance):
    x1, _, x2, _ = car_box(CAMERA, distance)
    # The widest part of the box is its rear face, closest to the camera.
    assert CAMERA.width_distance(x2 - x1, 1.85) == pytest.approx(distance, rel=0.01)


def test_width_sigma_combines_pixel_and_width_uncertainty():
    sigma = CAMERA.width_distance_sigma(20.0, width_px=50.0, sigma_px=1.0, sigma_width_m=0.0, object_width_m=1.85)
    assert sigma == pytest.approx(20.0 * math.sqrt(2) / 50.0)
    no_px = CAMERA.width_distance_sigma(20.0, 50.0, 0.0, 0.185, 1.85)
    assert no_px == pytest.approx(2.0)


def test_bearing_and_lateral_offset():
    cam = PinholeCamera(640, 360, 60.0, 1.4, pitch_deg=0.0)
    u = cam.project([[20.0, 2.0, 0.0]])[0, 0]
    assert cam.bearing(u) == pytest.approx(math.atan2(2.0, 20.0))
    assert cam.lateral_offset(u, 20.0) == pytest.approx(2.0)
    # Positive to the right, as CARLA's radar azimuth.
    assert cam.bearing(cam.cx + 10) > 0


def test_from_config_uses_measured_height():
    from svs.config import CameraConfig

    cam = PinholeCamera.from_config(CameraConfig(), mount_height_m=1.55)
    assert (cam.width, cam.height, cam.fov_deg, cam.pitch_deg) == (640, 360, 60.0, -6.0)
    assert cam.mount_height_m == 1.55
