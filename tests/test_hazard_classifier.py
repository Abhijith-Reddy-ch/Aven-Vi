import pytest
from aven_vi.spatial.hazard_classifier import HazardClassifier

def test_elevation_classification():
    classifier = HazardClassifier()
    h = 1000

    # 1. Ground hazard: low box near bottom of image
    ground_box = (200, 700, 400, 950)
    assert classifier.classify_elevation(ground_box, frame_height=h) == "GROUND"

    # 2. Body hazard: full height standing object
    body_box = (300, 200, 500, 850)
    assert classifier.classify_elevation(body_box, frame_height=h) == "BODY"

    # 3. Head / Overhead hazard: top of frame, hanging down partially
    head_box = (100, 50, 400, 300)
    assert classifier.classify_elevation(head_box, frame_height=h) == "HEAD"

def test_edge_cases():
    classifier = HazardClassifier()
    assert classifier.classify_elevation((0, 0, 10, 10), frame_height=0) == "BODY"
