from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class ImageVelocity:
    """
    Image-space velocity representation in pixels per second.
    
    CRITICAL RESEARCH NOTE:
    IMAGE VELOCITY != PHYSICAL VELOCITY.
    vx, vy, and speed represent 2D pixel-plane displacements over time (px/s),
    NOT real-world metric velocity (m/s).
    """
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0

@dataclass
class TrackState:
    """
    State container for a tracked object in AVEN-VI.
    Updated in Phase 3A to incorporate spatial reasoning and calibrated distance.
    """
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]              # (x1, y1, x2, y2)
    centroid: Tuple[float, float]                # (cx, cy)
    first_seen: float                            # Timestamp when track was initialized
    last_seen: float                             # Timestamp of most recent observation
    age: int = 1                                 # Total frames track has existed
    missing_frames: int = 0                      # Consecutive frames track was absent
    trajectory_history: List[Tuple[float, float, float]] = field(default_factory=list) # [(t, cx, cy), ...]
    raw_velocity: ImageVelocity = field(default_factory=ImageVelocity)
    smoothed_velocity: ImageVelocity = field(default_factory=ImageVelocity)
    is_active: bool = True

    # --- Phase 3A Spatial Reasoning Attributes ---
    clock_position: str = "12:00"                # e.g., "12:00", "11:00", "1:00", "9:00", "3:00"
    bearing_degrees: float = 0.0                 # Horizontal angle relative to optical axis (-35° to +35°)
    elevation_zone: str = "BODY"                 # "GROUND", "BODY", "HEAD"
    estimated_distance_m: float = 0.0            # Calibrated metric distance in meters
    in_corridor: bool = False                    # True if object impinges on central travel corridor
