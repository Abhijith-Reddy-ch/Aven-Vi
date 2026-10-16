from typing import Tuple

def calculate_centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Computes the 2D bounding-box centroid in image pixel coordinates.

    Args:
        bbox: (x1, y1, x2, y2) bounding box coordinates.

    Returns:
        (cx, cy): Centroid center point in pixel coordinates.
    """
    x1, y1, x2, y2 = bbox
    cx = (float(x1) + float(x2)) / 2.0
    cy = (float(y1) + float(y2)) / 2.0
    return (cx, cy)
