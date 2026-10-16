import pytest
from aven_vi.motion.centroid import calculate_centroid

def test_centroid_standard():
    bbox = (100, 100, 200, 200)
    cx, cy = calculate_centroid(bbox)
    assert cx == pytest.approx(150.0)
    assert cy == pytest.approx(150.0)

def test_centroid_asymmetric():
    bbox = (0, 10, 100, 50)
    cx, cy = calculate_centroid(bbox)
    assert cx == pytest.approx(50.0)
    assert cy == pytest.approx(30.0)

def test_centroid_floating_point():
    bbox = (10.5, 20.5, 30.5, 40.5)
    cx, cy = calculate_centroid(bbox)
    assert cx == pytest.approx(20.5)
    assert cy == pytest.approx(30.5)

def test_centroid_single_point():
    bbox = (100, 100, 100, 100)
    cx, cy = calculate_centroid(bbox)
    assert cx == pytest.approx(100.0)
    assert cy == pytest.approx(100.0)
