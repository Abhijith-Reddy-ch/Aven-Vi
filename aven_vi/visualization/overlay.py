import cv2
import numpy as np
from typing import List, Tuple
from aven_vi.utils.types import TrackState

COLOR_PALETTE = [
    (0, 255, 0),    # Green
    (255, 128, 0),  # Cyan
    (0, 165, 255),  # Orange
    (255, 0, 128),  # Violet
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (128, 255, 0),  # Lime
    (0, 128, 255),  # Amber
    (255, 255, 0),  # Light Blue
    (180, 105, 255) # Pink
]

class TrackingVisualizer:
    """
    Renders bounding boxes, centroids, trajectory trails, image-space velocities,
    and Phase 3A spatial reasoning annotations (clock bearing, elevation zone, calibrated distance).
    """
    def __init__(
        self,
        draw_trajectory: bool = True,
        draw_velocity: bool = True,
        draw_centroid: bool = True,
        draw_clock_bearing: bool = True,
        draw_distance: bool = True,
        draw_corridor: bool = True,
        corridor_ratio: float = 0.30,
        box_thickness: int = 2,
        font_scale: float = 0.50
    ):
        self.draw_trajectory = draw_trajectory
        self.draw_velocity = draw_velocity
        self.draw_centroid = draw_centroid
        self.draw_clock_bearing = draw_clock_bearing
        self.draw_distance = draw_distance
        self.draw_corridor = draw_corridor
        self.corridor_ratio = corridor_ratio
        self.box_thickness = box_thickness
        self.font_scale = font_scale

    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        return COLOR_PALETTE[track_id % len(COLOR_PALETTE)]

    def render(
        self,
        frame: np.ndarray,
        tracks: List[TrackState],
        fps: float = 0.0,
        depth_map: np.ndarray = None
    ) -> np.ndarray:
        """
        Draws visual annotations onto the frame.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # 1. Draw Central Walking Corridor Guide Lines
        if self.draw_corridor and w > 0:
            c_min = int((0.5 - (self.corridor_ratio / 2.0)) * w)
            c_max = int((0.5 + (self.corridor_ratio / 2.0)) * w)
            # Semi-transparent dashed vertical lines for corridor
            overlay = annotated.copy()
            cv2.line(overlay, (c_min, 0), (c_min, h), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(overlay, (c_max, 0), (c_max, h), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, "PATHWAY", (c_min + 10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

        # 2. Render Each Track
        for track in tracks:
            # Highlight obstacles in corridor with prominent color
            color = (0, 0, 255) if track.in_corridor else self._get_color(track.track_id)
            x1, y1, x2, y2 = track.bbox
            cx, cy = int(track.centroid[0]), int(track.centroid[1])

            # Draw Trajectory History Trail
            if self.draw_trajectory and len(track.trajectory_history) > 1:
                pts = np.array(
                    [[int(pt[1]), int(pt[2])] for pt in track.trajectory_history],
                    dtype=np.int32
                ).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

            # Draw Bounding Box (thickened if in direct corridor)
            thick = self.box_thickness + 1 if track.in_corridor else self.box_thickness
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

            # Draw Centroid Marker
            if self.draw_centroid:
                cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

            # Build Multi-Line Text Label
            # Line 1: Class, ID, Elevation zone
            path_flag = " [IN PATH]" if track.in_corridor else ""
            line1 = f"{track.class_name.upper()} #{track.track_id} [{track.elevation_zone}]{path_flag}"

            # Line 2: Spatial Bearing (Clock) + Distance in meters
            line2_parts = []
            if self.draw_clock_bearing:
                line2_parts.append(f"{track.clock_position} ({track.bearing_degrees:+.0f}\xb0)")
            if self.draw_distance:
                line2_parts.append(f"{track.estimated_distance_m:.1f}m")
            line2 = " | ".join(line2_parts) if line2_parts else f"Conf: {track.confidence:.2f}"

            # Line 3: Image-space velocity
            lines = [line1, line2]
            if self.draw_velocity:
                vx = track.smoothed_velocity.vx
                vy = track.smoothed_velocity.vy
                lines.append(f"V: ({vx:.0f}, {vy:.0f}) px/s")

            # Draw Text with background box
            line_height = int(18 * (self.font_scale / 0.5))
            text_y_start = max(y1 - (len(lines) * line_height) - 4, 15)

            for idx, line in enumerate(lines):
                ly = text_y_start + (idx * line_height)
                cv2.putText(annotated, line, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated, line, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Mini Depth PiP Radar (if depth_map provided)
        if depth_map is not None:
            disp_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_cmap = cv2.applyColorMap(disp_norm, cv2.COLORMAP_INFERNO)
            pip_h, pip_w = int(h * 0.22), int(w * 0.22)
            pip_resized = cv2.resize(depth_cmap, (pip_w, pip_h))
            annotated[15 : 15 + pip_h, w - pip_w - 15 : w - 15] = pip_resized
            cv2.rectangle(annotated, (w - pip_w - 15, 15), (w - 15, 15 + pip_h), (255, 255, 255), 1)

        # 4. Global Telemetry Overlay
        corridor_threats = sum(1 for t in tracks if t.in_corridor)
        hud_text = f"AVEN-VI Phase 3A | Tracks: {len(tracks)} (In Path: {corridor_threats}) | FPS: {fps:.1f}"
        cv2.putText(annotated, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

        return annotated
