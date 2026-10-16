from typing import Dict, List, Tuple, Optional
import numpy as np
from aven_vi.utils.types import TrackState, ImageVelocity
from aven_vi.motion.centroid import calculate_centroid
from aven_vi.motion.trajectory import TrajectoryHistory
from aven_vi.motion.motion_estimator import MotionEstimator
from aven_vi.perception.detector import ObstacleDetector
from aven_vi.spatial.clock_mapper import ClockMapper
from aven_vi.spatial.hazard_classifier import HazardClassifier
from aven_vi.spatial.distance_calibrator import DistanceCalibrator

class ByteTrackLifecycleManager:
    """
    Orchestrates ByteTrack multi-object tracking, temporal motion estimation,
    track lifecycle management, and Phase 3A spatial reasoning (clock position, elevation, calibrated distance).
    """
    def __init__(
        self,
        detector: ObstacleDetector,
        ema_alpha: float = 0.20,
        max_history: int = 30,
        max_missing_frames: int = 15,
        clock_mapper: Optional[ClockMapper] = None,
        hazard_classifier: Optional[HazardClassifier] = None,
        distance_calibrator: Optional[DistanceCalibrator] = None
    ):
        self.detector = detector
        self.max_history = int(max_history)
        self.max_missing_frames = int(max_missing_frames)
        self.motion_estimator = MotionEstimator(ema_alpha=ema_alpha)

        # Spatial Reasoning Components
        self.clock_mapper = clock_mapper or ClockMapper()
        self.hazard_classifier = hazard_classifier or HazardClassifier()
        self.distance_calibrator = distance_calibrator or DistanceCalibrator()

        # Active tracks dictionary: track_id -> TrackState
        self.active_tracks: Dict[int, TrackState] = {}
        # Trajectory managers: track_id -> TrajectoryHistory
        self.trajectories: Dict[int, TrajectoryHistory] = {}

    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        depth_estimator: Optional[any] = None
    ) -> List[TrackState]:
        """
        Processes a video frame with ByteTrack, computes spatial bearing, elevation,
        calibrated distance, and updates active tracks.

        Args:
            frame: BGR video frame
            timestamp: current frame timestamp (seconds)
            depth_estimator: optional MonocularDepthEstimator instance

        Returns:
            List of TrackState objects detected and updated in the current frame.
        """
        h_frame, w_frame = frame.shape[:2]

        # Execute YOLO tracking with ByteTrack
        results = self.detector.model.track(
            frame,
            classes=self.detector.target_class_ids,
            conf=self.detector.confidence,
            persist=True,
            verbose=False
        )[0]

        detected_track_ids = set()
        current_frame_tracks: List[TrackState] = []

        if results.boxes is not None and len(results.boxes) > 0 and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy()

            for i, tid_raw in enumerate(track_ids):
                track_id = int(tid_raw)
                detected_track_ids.add(track_id)

                cls_id = int(classes[i])
                cls_name = self.detector.get_class_name(cls_id)
                conf = float(confs[i])
                bbox = tuple(map(int, boxes[i]))  # (x1, y1, x2, y2)
                cx, cy = calculate_centroid(bbox)

                # --- Spatial Reasoning Attributes ---
                clock_pos, bearing_deg = self.clock_mapper.calculate_bearing(cx, frame_width=w_frame)
                in_corridor = self.clock_mapper.is_in_corridor(bbox, frame_width=w_frame)
                elevation = self.hazard_classifier.classify_elevation(bbox, frame_height=h_frame)

                # Calibrated distance
                median_disp = depth_estimator.sample_roi_disparity(bbox) if depth_estimator else None
                dist_m = self.distance_calibrator.calibrate(median_disp, bbox, frame_height=h_frame)

                # Trajectory update
                if track_id not in self.trajectories:
                    self.trajectories[track_id] = TrajectoryHistory(max_history=self.max_history)
                traj = self.trajectories[track_id]
                traj.append(timestamp, cx, cy)

                if track_id in self.active_tracks:
                    # Update existing track
                    prev_state = self.active_tracks[track_id]
                    dt = timestamp - prev_state.last_seen
                    prev_centroid = prev_state.centroid
                    prev_smoothed_vel = prev_state.smoothed_velocity

                    raw_vel, smoothed_vel = self.motion_estimator.calculate_velocity(
                        current_centroid=(cx, cy),
                        previous_centroid=prev_centroid,
                        dt=dt,
                        previous_smoothed_velocity=prev_smoothed_vel
                    )

                    state = TrackState(
                        track_id=track_id,
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=bbox,
                        centroid=(cx, cy),
                        first_seen=prev_state.first_seen,
                        last_seen=timestamp,
                        age=prev_state.age + 1,
                        missing_frames=0,
                        trajectory_history=traj.get_points(),
                        raw_velocity=raw_vel,
                        smoothed_velocity=smoothed_vel,
                        is_active=True,
                        clock_position=clock_pos,
                        bearing_degrees=round(bearing_deg, 1),
                        elevation_zone=elevation,
                        estimated_distance_m=dist_m,
                        in_corridor=in_corridor
                    )
                else:
                    # New track initialized
                    zero_vel = ImageVelocity(vx=0.0, vy=0.0, speed=0.0)
                    state = TrackState(
                        track_id=track_id,
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=bbox,
                        centroid=(cx, cy),
                        first_seen=timestamp,
                        last_seen=timestamp,
                        age=1,
                        missing_frames=0,
                        trajectory_history=traj.get_points(),
                        raw_velocity=zero_vel,
                        smoothed_velocity=zero_vel,
                        is_active=True,
                        clock_position=clock_pos,
                        bearing_degrees=round(bearing_deg, 1),
                        elevation_zone=elevation,
                        estimated_distance_m=dist_m,
                        in_corridor=in_corridor
                    )

                self.active_tracks[track_id] = state
                current_frame_tracks.append(state)

        # Lifecycle Management: Purge expired tracks
        expired_track_ids = []
        for tid, state in self.active_tracks.items():
            if tid not in detected_track_ids:
                state.missing_frames += 1
                state.is_active = False
                if state.missing_frames > self.max_missing_frames:
                    expired_track_ids.append(tid)

        for tid in expired_track_ids:
            del self.active_tracks[tid]
            if tid in self.trajectories:
                del self.trajectories[tid]

        return current_frame_tracks
