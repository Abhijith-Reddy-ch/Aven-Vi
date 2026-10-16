from typing import Tuple

class ClockMapper:
    """
    Translates 2D image coordinates into human-intuitive clock-face directions (9:00 - 3:00)
    and checks if an obstacle occupies the pedestrian walking corridor.
    """
    def __init__(self, hfov_deg: float = 70.0, corridor_ratio: float = 0.30):
        self.hfov_deg = float(hfov_deg)
        self.corridor_ratio = float(corridor_ratio)

    def calculate_bearing(self, cx: float, frame_width: int) -> Tuple[str, float]:
        """
        Calculates the clock position and horizontal bearing angle (degrees) for a centroid.

        Args:
            cx: Centroid X coordinate in pixels.
            frame_width: Total image width in pixels.

        Returns:
            (clock_str, bearing_degrees)
        """
        if frame_width <= 0:
            return "12:00", 0.0

        u = cx / float(frame_width)
        # Center offset: -0.5 (left edge) to +0.5 (right edge)
        center_offset = u - 0.5
        bearing_deg = center_offset * self.hfov_deg

        # Sector thresholds based on bearing angle
        if bearing_deg < -25.0:
            clock_pos = "9:00"
        elif bearing_deg < -15.0:
            clock_pos = "10:00"
        elif bearing_deg < -5.0:
            clock_pos = "11:00"
        elif bearing_deg <= 5.0:
            clock_pos = "12:00"
        elif bearing_deg <= 15.0:
            clock_pos = "1:00"
        elif bearing_deg <= 25.0:
            clock_pos = "2:00"
        else:
            clock_pos = "3:00"

        return clock_pos, bearing_deg

    def is_in_corridor(self, bbox: Tuple[int, int, int, int], frame_width: int) -> bool:
        """
        Checks whether the bounding box intersects the central walking corridor.
        Corridor spans [0.5 - corridor_ratio/2, 0.5 + corridor_ratio/2] * frame_width.
        """
        if frame_width <= 0:
            return False

        x1, y1, x2, y2 = bbox
        corridor_min = (0.5 - (self.corridor_ratio / 2.0)) * frame_width
        corridor_max = (0.5 + (self.corridor_ratio / 2.0)) * frame_width

        # Check interval overlap [x1, x2] and [corridor_min, corridor_max]
        return not (x2 < corridor_min or x1 > corridor_max)
