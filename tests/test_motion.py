import pytest
import math
from aven_vi.motion.motion_estimator import MotionEstimator
from aven_vi.utils.types import ImageVelocity

def test_missing_previous_observation():
    """First frame observation with no previous history must return 0 velocity."""
    estimator = MotionEstimator(ema_alpha=0.20)
    raw, smoothed = estimator.calculate_velocity(
        current_centroid=(150.0, 150.0),
        previous_centroid=None,
        dt=0.033
    )
    assert raw.vx == 0.0
    assert raw.vy == 0.0
    assert raw.speed == 0.0
    assert smoothed.vx == 0.0
    assert smoothed.vy == 0.0
    assert smoothed.speed == 0.0

def test_zero_movement():
    """Object at stationary position over elapsed time."""
    estimator = MotionEstimator(ema_alpha=0.20)
    raw, smoothed = estimator.calculate_velocity(
        current_centroid=(100.0, 100.0),
        previous_centroid=(100.0, 100.0),
        dt=0.1
    )
    assert raw.vx == pytest.approx(0.0)
    assert raw.vy == pytest.approx(0.0)
    assert raw.speed == pytest.approx(0.0)
    assert smoothed.vx == pytest.approx(0.0)
    assert smoothed.vy == pytest.approx(0.0)
    assert smoothed.speed == pytest.approx(0.0)

def test_horizontal_movement():
    """?cx = 100 px, ?cy = 0 px, dt = 1.0 s -> vx = 100 px/s, vy = 0, speed = 100 px/s."""
    estimator = MotionEstimator(ema_alpha=0.20)
    raw, smoothed = estimator.calculate_velocity(
        current_centroid=(200.0, 100.0),
        previous_centroid=(100.0, 100.0),
        dt=1.0
    )
    assert raw.vx == pytest.approx(100.0)
    assert raw.vy == pytest.approx(0.0)
    assert raw.speed == pytest.approx(100.0)
    # First velocity initializes smoothed velocity directly
    assert smoothed.vx == pytest.approx(100.0)
    assert smoothed.vy == pytest.approx(0.0)
    assert smoothed.speed == pytest.approx(100.0)

def test_vertical_movement():
    """?cx = 0 px, ?cy = -50 px, dt = 0.5 s -> vx = 0, vy = -100 px/s, speed = 100 px/s."""
    estimator = MotionEstimator(ema_alpha=0.20)
    raw, smoothed = estimator.calculate_velocity(
        current_centroid=(100.0, 50.0),
        previous_centroid=(100.0, 100.0),
        dt=0.5
    )
    assert raw.vx == pytest.approx(0.0)
    assert raw.vy == pytest.approx(-100.0)
    assert raw.speed == pytest.approx(100.0)
    assert smoothed.vx == pytest.approx(0.0)
    assert smoothed.vy == pytest.approx(-100.0)
    assert smoothed.speed == pytest.approx(100.0)

def test_diagonal_movement():
    """?cx = 30 px, ?cy = 40 px, dt = 1.0 s -> vx = 30, vy = 40, speed = 50 (3-4-5 triangle)."""
    estimator = MotionEstimator(ema_alpha=0.20)
    raw, smoothed = estimator.calculate_velocity(
        current_centroid=(130.0, 140.0),
        previous_centroid=(100.0, 100.0),
        dt=1.0
    )
    assert raw.vx == pytest.approx(30.0)
    assert raw.vy == pytest.approx(40.0)
    assert raw.speed == pytest.approx(50.0)

def test_ema_smoothing_step_response():
    """
    Verifies that EMA filter dampens velocity spikes:
    Previous smoothed = 100 px/s
    Raw spike = 200 px/s
    With alpha = 0.20:
    Smoothed = 0.20 * 200 + 0.80 * 100 = 40 + 80 = 120 px/s
    """
    alpha = 0.20
    estimator = MotionEstimator(ema_alpha=alpha)
    prev_smoothed = ImageVelocity(vx=100.0, vy=0.0, speed=100.0)

    # Move 200 px in 1 second
    raw, smoothed = estimator.calculate_velocity(
        current_centroid=(300.0, 100.0),
        previous_centroid=(100.0, 100.0),
        dt=1.0,
        previous_smoothed_velocity=prev_smoothed
    )

    assert raw.vx == pytest.approx(200.0)
    expected_smooth_vx = (200.0 * alpha) + (100.0 * (1.0 - alpha))
    assert smoothed.vx == pytest.approx(expected_smooth_vx)
    assert smoothed.vx == pytest.approx(120.0)
    assert smoothed.speed == pytest.approx(120.0)
