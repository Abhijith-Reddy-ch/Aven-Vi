import pytest
from aven_vi.spatial.distance_calibrator import DistanceCalibrator

def test_disparity_distance_scaling():
    calibrator = DistanceCalibrator(disparity_scale=350.0, disparity_offset=0.5)

    # High disparity (near obstacle) -> short distance in meters
    near_dist = calibrator.estimate_from_disparity(median_disparity=300.0)
    assert 0.8 <= near_dist <= 1.5

    # Low disparity (distant obstacle) -> large distance in meters
    far_dist = calibrator.estimate_from_disparity(median_disparity=35.0)
    assert 7.0 <= far_dist <= 12.0

    # Near-zero disparity clamped to max 25m
    infinite_dist = calibrator.estimate_from_disparity(median_disparity=0.0)
    assert infinite_dist <= 25.0

def test_ground_contact_distance():
    calibrator = DistanceCalibrator(camera_height_m=1.35)
    h = 1000

    # Obstacle touching bottom of frame (y2=980) -> close distance
    close_contact = calibrator.estimate_from_ground_contact((100, 500, 300, 980), frame_height=h)
    assert close_contact is not None
    assert 0.5 <= close_contact <= 3.0

    # Obstacle touching horizon (y2=500) -> not considered ground contact
    horizon_box = calibrator.estimate_from_ground_contact((100, 100, 300, 500), frame_height=h)
    assert horizon_box is None

def test_combined_calibration():
    calibrator = DistanceCalibrator()
    # Fused estimate produces smooth float in meters
    dist = calibrator.calibrate(median_disparity=150.0, bbox=(100, 400, 300, 800), frame_height=1000)
    assert isinstance(dist, float)
    assert 0.5 <= dist <= 15.0
