import math
from typing import Tuple, Optional
from aven_vi.utils.types import ImageVelocity

class MotionEstimator:
    """
    Estimates 2D image-space velocity and applies Exponential Moving Average (EMA) smoothing.
    
    IMPORTANT RESEARCH INTEGRITY NOTE:
    IMAGE VELOCITY != PHYSICAL VELOCITY.
    Calculations here operate purely in pixel-space coordinates (pixels per second).
    They do NOT represent physical speed in meters/second.
    """
    def __init__(self, ema_alpha: float = 0.20):
        self.ema_alpha = float(ema_alpha)

    def calculate_velocity(
        self,
        current_centroid: Tuple[float, float],
        previous_centroid: Optional[Tuple[float, float]],
        dt: float,
        previous_smoothed_velocity: Optional[ImageVelocity] = None
    ) -> Tuple[ImageVelocity, ImageVelocity]:
        """
        Calculates raw and EMA-smoothed image velocity in px/s.

        Args:
            current_centroid: (cx, cy) at current timestamp
            previous_centroid: (prev_cx, prev_cy) at previous timestamp, or None
            dt: elapsed time delta in seconds (?t)
            previous_smoothed_velocity: previously smoothed ImageVelocity, or None

        Returns:
            (raw_velocity, smoothed_velocity)
        """
        # Case 1: First observation or invalid time delta -> zero velocity
        if previous_centroid is None or dt <= 0.0:
            zero_vel = ImageVelocity(vx=0.0, vy=0.0, speed=0.0)
            return zero_vel, zero_vel

        # Finite difference displacement
        cx, cy = current_centroid
        prev_cx, prev_cy = previous_centroid
        dcx = cx - prev_cx
        dcy = cy - prev_cy

        raw_vx = dcx / dt
        raw_vy = dcy / dt
        raw_speed = math.sqrt(raw_vx * raw_vx + raw_vy * raw_vy)
        raw_vel = ImageVelocity(vx=raw_vx, vy=raw_vy, speed=raw_speed)

        # Apply EMA smoothing
        if previous_smoothed_velocity is None or (
            previous_smoothed_velocity.vx == 0.0 and previous_smoothed_velocity.vy == 0.0
        ):
            # First velocity estimate initializes the smoother
            smoothed_vel = ImageVelocity(vx=raw_vx, vy=raw_vy, speed=raw_speed)
        else:
            smoothed_vx = (raw_vx * self.ema_alpha) + (previous_smoothed_velocity.vx * (1.0 - self.ema_alpha))
            smoothed_vy = (raw_vy * self.ema_alpha) + (previous_smoothed_velocity.vy * (1.0 - self.ema_alpha))
            smoothed_speed = math.sqrt(smoothed_vx * smoothed_vx + smoothed_vy * smoothed_vy)
            smoothed_vel = ImageVelocity(vx=smoothed_vx, vy=smoothed_vy, speed=smoothed_speed)

        return raw_vel, smoothed_vel
