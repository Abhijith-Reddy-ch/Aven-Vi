from typing import Tuple

class HazardClassifier:
    """
    Classifies detected obstacles into vertical elevation zones:
    - GROUND: Tripping hazards, curbs, steps, low obstacles
    - BODY: Direct collision threats (pedestrians, vehicles, bollards, doors)
    - HEAD: Overhead hazards (low tree branches, awnings, signage)
    """
    def __init__(self, ground_ratio: float = 0.65, head_ratio: float = 0.25):
        self.ground_ratio = float(ground_ratio)
        self.head_ratio = float(head_ratio)

    def classify_elevation(self, bbox: Tuple[int, int, int, int], frame_height: int) -> str:
        """
        Assigns an elevation hazard label based on bounding box vertical coordinates.

        Args:
            bbox: (x1, y1, x2, y2)
            frame_height: Total image height in pixels.

        Returns:
            "GROUND", "BODY", or "HEAD"
        """
        if frame_height <= 0:
            return "BODY"

        x1, y1, x2, y2 = bbox
        norm_y1 = y1 / float(frame_height)
        norm_y2 = y2 / float(frame_height)

        # 1. Head / Overhead Hazard: Starts high and does not touch the ground
        if norm_y1 <= self.head_ratio and norm_y2 < 0.65:
            return "HEAD"

        # 2. Ground Hazard: Entire object is confined to the lower portion of the frame
        if norm_y1 >= 0.50 and norm_y2 >= self.ground_ratio:
            return "GROUND"

        # 3. Body / Torso Level: Default for full-body obstacles, vehicles, pedestrians
        return "BODY"
