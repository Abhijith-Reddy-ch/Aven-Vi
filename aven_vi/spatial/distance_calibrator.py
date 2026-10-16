import numpy as np
from typing import Tuple, Optional

class DistanceCalibrator:
    """
    Translates monocular depth/disparity extractions and bounding box ground-contacts
    into calibrated metric distance estimates (meters).
    
    RESEARCH INTEGRITY NOTE:
    Monocular depth is inherently subject to scale ambiguity without stereo or LiDAR ground truth.
    Estimates produced here are calibrated metric approximations for assistive navigation,
    NOT certified survey-grade metrology.
    """
    def __init__(
        self,
        camera_height_m: float = 1.35,
        vfov_deg: float = 45.0,
        disparity_scale: float = 350.0,
        disparity_offset: float = 0.5
    ):
        self.camera_height_m = float(camera_height_m)
        self.vfov_deg = float(vfov_deg)
        self.disparity_scale = float(disparity_scale)
        self.disparity_offset = float(disparity_offset)

    def estimate_from_disparity(self, median_disparity: float) -> float:
        """
        Calibrates MiDaS relative disparity into approximate meters:
        d = scale / (disparity + offset)
        """
        disp = max(0.01, float(median_disparity))
        # When disparity is high (near camera), distance is short (~1.0 m)
        # When disparity is low (far), distance is large (~8-15 m)
        dist_m = self.disparity_scale / (disp + self.disparity_offset)
        # Bound distance to realistic pedestrian situational awareness bubble [0.3m, 25.0m]
        return float(np.clip(dist_m, 0.3, 25.0))

    def estimate_from_ground_contact(
        self,
        bbox: Tuple[int, int, int, int],
        frame_height: int,
        pitch_deg: float = 5.0
    ) -> Optional[float]:
        """
        Estimates ground distance using pinhole geometry assuming obstacle base touches the floor.
        """
        if frame_height <= 0:
            return None

        _, _, _, y2 = bbox
        norm_y = y2 / float(frame_height)  # 0.0 at top, 1.0 at bottom

        # Ground plane is typically in lower half (norm_y > 0.5)
        if norm_y <= 0.52:
            return None

        # Angle below horizon
        angle_rad = np.radians(pitch_deg + (norm_y - 0.5) * self.vfov_deg)
        if angle_rad <= 0.05:
            return None

        dist_m = self.camera_height_m / np.tan(angle_rad)
        return float(np.clip(dist_m, 0.3, 25.0))

    def calibrate(
        self,
        median_disparity: Optional[float],
        bbox: Tuple[int, int, int, int],
        frame_height: int
    ) -> float:
        """
        Combines monocular disparity and geometric heuristics into a final distance estimate (meters).
        """
        dist_disp = self.estimate_from_disparity(median_disparity) if median_disparity is not None else None
        dist_ground = self.estimate_from_ground_contact(bbox, frame_height)

        if dist_disp is not None and dist_ground is not None:
            # Weighted fusion: disparity (60%) + ground geometry (40%)
            fused_dist = (0.60 * dist_disp) + (0.40 * dist_ground)
            return round(fused_dist, 2)
        elif dist_disp is not None:
            return round(dist_disp, 2)
        elif dist_ground is not None:
            return round(dist_ground, 2)
        else:
            return 3.0  # Conservative mid-range fallback
