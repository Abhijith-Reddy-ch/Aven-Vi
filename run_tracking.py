import argparse
import os
import time
import cv2
import numpy as np

from aven_vi.config.config_loader import load_config
from aven_vi.perception.detector import ObstacleDetector
from aven_vi.perception.tracker import ByteTrackLifecycleManager
from aven_vi.perception.depth_estimator import MonocularDepthEstimator
from aven_vi.spatial.clock_mapper import ClockMapper
from aven_vi.spatial.hazard_classifier import HazardClassifier
from aven_vi.spatial.distance_calibrator import DistanceCalibrator
from aven_vi.visualization.overlay import TrackingVisualizer
from aven_vi.audio.audio_coordinator import AudioCoordinator

def run_pipeline(
    input_video: str,
    output_video: str = "aven_vi_spatial_output.mp4",
    config_path: str = "config/av_config.yaml",
    display: bool = False,
    no_audio: bool = False,
    audio_mode: str = None,
    mock_audio: bool = False
):
    print("=" * 65)
    print("AVEN-VI Phase 3: Spatial Reasoning, Depth & Audio Guidance")
    print("=" * 65)

    # 1. Load Configuration
    cfg = load_config(config_path)
    det_cfg = cfg.get("detection", {})
    mot_cfg = cfg.get("motion", {})
    dep_cfg = cfg.get("depth", {})
    spa_cfg = cfg.get("spatial", {})
    aud_cfg = cfg.get("audio", {})
    vis_cfg = cfg.get("visualization", {})

    # 2. Initialize Spatial Reasoning Subsystems
    clock_mapper = ClockMapper(
        hfov_deg=spa_cfg.get("hfov_deg", 70.0),
        corridor_ratio=spa_cfg.get("corridor_ratio", 0.30)
    )
    hazard_classifier = HazardClassifier()
    distance_calibrator = DistanceCalibrator(
        camera_height_m=spa_cfg.get("camera_height_m", 1.35),
        vfov_deg=spa_cfg.get("vfov_deg", 45.0),
        disparity_scale=dep_cfg.get("disparity_scale", 350.0),
        disparity_offset=dep_cfg.get("disparity_offset", 0.5)
    )

    # 3. Initialize Audio Guidance Subsystem
    audio_enabled = aud_cfg.get("enabled", True) if not no_audio else False
    selected_mode = audio_mode if audio_mode is not None else aud_cfg.get("mode", "speech")
    is_mock_audio = mock_audio or aud_cfg.get("mock_mode", False)

    audio_coordinator = AudioCoordinator(
        enabled=audio_enabled,
        mode=selected_mode,
        speech_rate=aud_cfg.get("speech_rate", 175),
        volume=aud_cfg.get("volume", 0.90),
        alert_cooldown_s=aud_cfg.get("alert_cooldown_s", 3.5),
        clear_pathway_interval_s=aud_cfg.get("clear_pathway_interval_s", 6.0),
        mock_mode=is_mock_audio
    )

    # 4. Initialize Perception & Depth
    detector = ObstacleDetector(
        model_name=det_cfg.get("model", "yolov8n.pt"),
        confidence=det_cfg.get("confidence", 0.35),
        target_class_names=det_cfg.get("classes", None)
    )

    depth_estimator = None
    if dep_cfg.get("enabled", True):
        depth_estimator = MonocularDepthEstimator(
            model_type=dep_cfg.get("model", "MiDaS_small")
        )

    tracker = ByteTrackLifecycleManager(
        detector=detector,
        ema_alpha=mot_cfg.get("ema_alpha", 0.20),
        max_history=mot_cfg.get("max_history", 30),
        max_missing_frames=mot_cfg.get("max_missing_frames", 15),
        clock_mapper=clock_mapper,
        hazard_classifier=hazard_classifier,
        distance_calibrator=distance_calibrator
    )

    visualizer = TrackingVisualizer(
        draw_trajectory=vis_cfg.get("draw_trajectory", True),
        draw_velocity=vis_cfg.get("draw_velocity", True),
        draw_centroid=vis_cfg.get("draw_centroid", True),
        draw_clock_bearing=vis_cfg.get("draw_clock_bearing", True),
        draw_distance=vis_cfg.get("draw_distance", True),
        draw_corridor=vis_cfg.get("draw_corridor", True),
        corridor_ratio=spa_cfg.get("corridor_ratio", 0.30),
        box_thickness=vis_cfg.get("box_thickness", 2),
        font_scale=vis_cfg.get("font_scale", 0.50)
    )

    # 5. Open Video Stream
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"[ERROR] Cannot open input video: {input_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0:
        fps_in = 30.0

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        (width, height)
    )

    print(f"[INFO] Input: {input_video} ({width}x{height} @ {fps_in:.1f} FPS)")
    print(f"[INFO] Output: {output_video}")
    print(f"[INFO] Audio: Enabled={audio_enabled}, Mode={selected_mode}, Mock={is_mock_audio}")
    print("[INFO] Processing frames with Spatial Reasoning, Depth & Non-blocking Audio...")

    depth_stride = max(1, int(dep_cfg.get("depth_stride", 1)))
    frame_count = 0
    total_latency_ms = 0.0
    start_time = time.time()
    all_tracked_ids = set()

    try:
        while cap.isOpened():
            t_frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            timestamp = time.time()

            # Compute Depth Map (respecting depth_stride)
            depth_map = None
            if depth_estimator is not None and (frame_count % depth_stride == 0 or depth_estimator.latest_disparity_map is None):
                depth_map = depth_estimator.estimate_disparity(frame)
            elif depth_estimator is not None:
                depth_map = depth_estimator.latest_disparity_map

            # Update Tracking & Spatial Pipeline
            active_tracks = tracker.update(
                frame,
                timestamp=timestamp,
                depth_estimator=depth_estimator
            )
            for t in active_tracks:
                all_tracked_ids.add(t.track_id)

            # Asynchronous, Non-blocking Audio Guidance
            if audio_enabled:
                audio_coordinator.process_frame(active_tracks, timestamp=timestamp)

            # Measure Latency
            latency_ms = (time.time() - t_frame_start) * 1000.0
            total_latency_ms += latency_ms
            current_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

            # Render Annotations
            annotated_frame = visualizer.render(
                frame,
                active_tracks,
                fps=current_fps,
                depth_map=depth_map
            )
            out.write(annotated_frame)

            if display:
                cv2.imshow("AVEN-VI Assistive Navigation", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()
        audio_coordinator.stop()

    elapsed = time.time() - start_time
    avg_latency = (total_latency_ms / frame_count) if frame_count > 0 else 0.0
    effective_fps = (frame_count / elapsed) if elapsed > 0 else 0.0
    alerts_dispatched = len(audio_coordinator.dispatcher.spoken_log)

    print("\n" + "=" * 65)
    print("AVEN-VI Phase 3 - Execution Summary")
    print("=" * 65)
    print(f"Total Frames Processed : {frame_count}")
    print(f"Total Wall-Clock Time  : {elapsed:.2f} s")
    print(f"Mean Latency / Frame   : {avg_latency:.1f} ms")
    print(f"Effective Throughput   : {effective_fps:.1f} FPS")
    print(f"Unique Track IDs Seen  : {len(all_tracked_ids)}")
    print(f"Active Tracks at Exit  : {len(tracker.active_tracks)}")
    print(f"Audio Alerts Emitted   : {alerts_dispatched}")
    print(f"Saved Output Video     : {output_video}")
    print("=" * 65)

    return {
        "frames": frame_count,
        "mean_latency_ms": avg_latency,
        "effective_fps": effective_fps,
        "unique_tracks": len(all_tracked_ids),
        "alerts_dispatched": alerts_dispatched
    }

def main():
    parser = argparse.ArgumentParser(description="AVEN-VI Phase 3: Spatial Reasoning, Depth & Audio")
    parser.add_argument("--input", default="driving_video.mp4", help="Path to input video")
    parser.add_argument("--output", default="aven_vi_output.mp4", help="Path to output video")
    parser.add_argument("--config", default="config/av_config.yaml", help="Path to YAML configuration")
    parser.add_argument("--show", action="store_true", help="Display visualization window")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio alerts")
    parser.add_argument("--audio-mode", choices=["speech", "earcon", "hybrid"], default=None, help="Audio output mode")
    parser.add_argument("--mock-audio", action="store_true", help="Use mock audio dispatcher (no physical TTS playback)")
    args = parser.parse_args()

    run_pipeline(
        input_video=args.input,
        output_video=args.output,
        config_path=args.config,
        display=args.show,
        no_audio=args.no_audio,
        audio_mode=args.audio_mode,
        mock_audio=args.mock_audio
    )

if __name__ == "__main__":
    main()
