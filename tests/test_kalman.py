import numpy as np
import pytest

from svs.fusion.kalman import H_CAMERA, H_RADAR, ConstantVelocityKF, process_noise

DT = 0.05


def test_process_noise_shape_and_symmetry():
    Q = process_noise(DT, 2.0)
    assert Q.shape == (2, 2)
    assert np.allclose(Q, Q.T)
    assert np.all(np.linalg.eigvalsh(Q) >= -1e-15)
    assert Q[1, 1] == pytest.approx(4.0 * DT ** 2)


def test_predict_moves_gap_by_rate():
    kf = ConstantVelocityKF([20.0, -2.0], np.eye(2), sigma_accel=1.0)
    P_before = kf.P.copy()
    kf.predict(0.5)
    assert kf.gap == pytest.approx(19.0)
    assert kf.gap_rate == pytest.approx(-2.0)
    assert kf.P[0, 0] > P_before[0, 0]   # uncertainty grows without measurements


def test_precise_measurement_dominates():
    kf = ConstantVelocityKF([20.0, 0.0], np.eye(2) * 100, sigma_accel=1.0)
    kf.update([15.0, -1.0], H_RADAR, np.eye(2) * 1e-6)
    assert kf.gap == pytest.approx(15.0, abs=1e-3)
    assert kf.gap_rate == pytest.approx(-1.0, abs=1e-3)


def simulate(H, R, steps=600, seed=0, rate=-2.0):
    """Constant-velocity target, noisy measurements; returns (errors, NIS values)."""
    rng = np.random.default_rng(seed)
    truth = np.array([40.0, rate])
    kf = ConstantVelocityKF([40.0, 0.0], np.diag([4.0, 9.0]), sigma_accel=0.3)
    errors, nis = [], []
    for k in range(steps):
        truth = truth + np.array([truth[1] * DT, 0.0])
        if truth[0] < 5:   # keep the target in range: restart far away
            truth[0] = 40.0
            kf.x[0] += 35.0
        kf.predict(DT)
        z = H @ truth + rng.multivariate_normal(np.zeros(len(R)), R)
        nis.append(kf.update(z, H, R))
        errors.append(kf.x - truth)
    return np.array(errors[100:]), np.array(nis[100:])


def test_radar_filter_is_accurate_and_consistent():
    R = np.diag([0.3 ** 2, 0.2 ** 2])
    errors, nis = simulate(H_RADAR, R)
    assert np.sqrt(np.mean(errors[:, 0] ** 2)) < 0.3    # smoother than one measurement
    assert np.sqrt(np.mean(errors[:, 1] ** 2)) < 0.2
    # Consistency: NIS follows chi-square with 2 degrees of freedom -> mean ~2.
    assert 1.6 < nis.mean() < 2.5


def test_camera_only_filter_infers_closing_speed():
    errors, nis = simulate(H_CAMERA, np.array([[1.0]]))
    assert np.sqrt(np.mean(errors[:, 1] ** 2)) < 0.6    # rate never measured, only inferred
    assert 0.8 < nis.mean() < 1.25
