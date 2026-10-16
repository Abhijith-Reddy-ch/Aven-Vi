# Phase 2: Perception and Temporal Tracking Documentation

## Executive Overview
Phase 2 establishes the foundational spatial-temporal perception layer for **AVEN-VI** (Assistive Vision-Enabled Navigation for Visually Impaired Users). 
In this phase, the codebase was refactored from the monolithic automotive baseline into a modular, testable, and extensible architecture. The perception scope was expanded from vehicle-only detection to multi-class obstacle detection and persistent tracking with image-space motion dynamics and track lifecycle management.

---

## 1. Modular Architecture

```text
AVEN-VI/
├── config/
│   └── av_config.yaml                  # Centralized YAML configuration
├── aven_vi/
│   ├── __init__.py                     # Package declaration
│   ├── utils/
│   │   ├── __init__.py
│   │   └── types.py                    # Typed data structures (TrackState, ImageVelocity)
│   ├── config/
│   │   ├── __init__.py
│   │   └── config_loader.py            # YAML configuration loader & class ID resolver
│   ├── motion/
│   │   ├── __init__.py
│   │   ├── centroid.py                 # 2D centroid extraction
│   │   ├── trajectory.py               # Bounded FIFO trajectory history manager
│   │   └── motion_estimator.py         # 2D image-space velocity & EMA filter
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── detector.py                 # Configurable YOLOv8 multi-class detector
│   │   └── tracker.py                  # ByteTrack integration & lifecycle manager
│   └── visualization/
│       ├── __init__.py
│       └── overlay.py                  # Bounding box, trajectory, & velocity HUD
├── tests/
│   ├── test_centroid.py                # Unit tests for centroid math
│   ├── test_motion.py                  # Unit tests for image velocity & EMA smoothing
│   └── test_trajectory.py              # Unit tests for trajectory bounds & track eviction
├── run_tracking.py                     # Phase 2 pipeline execution entry point
└── visionfusion_adas.py                # Original baseline preserved intact
```

---

## 2. Supported Object Classes & Centralized Configuration

Rather than hard-coding integer class IDs throughout the codebase, AVEN-VI centralizes all detection and motion parameters in `config/av_config.yaml`:

```yaml
detection:
  model: "yolov8n.pt"
  confidence: 0.35
  classes:
    - person
    - bicycle
    - car
    - motorcycle
    - bus
    - truck
    - dog
    - chair
    - bench
    - traffic light
    - stop sign

motion:
  ema_alpha: 0.20
  max_history: 30
  max_missing_frames: 15

visualization:
  draw_trajectory: true
  draw_velocity: true
  draw_centroid: true
  box_thickness: 2
  font_scale: 0.50
```

The `resolve_target_class_ids()` function dynamically queries the YOLO model's vocabulary at boot time and maps class names to model indices, completely removing hardcoded class arrays.

---

## 3. Data Structures & State Management

Each tracked object is maintained in a strongly-typed `TrackState` dataclass:

```python
@dataclass
class ImageVelocity:
    vx: float = 0.0      # Horizontal velocity (pixels/second)
    vy: float = 0.0      # Vertical velocity (pixels/second)
    speed: float = 0.0   # Euclidean 2D speed (pixels/second)

@dataclass
class TrackState:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]              # (x1, y1, x2, y2)
    centroid: Tuple[float, float]                # (cx, cy)
    first_seen: float                            # Initial frame timestamp
    last_seen: float                             # Most recent frame timestamp
    age: int = 1                                 # Total frames track has existed
    missing_frames: int = 0                      # Consecutive frames absent
    trajectory_history: List[Tuple[float, float, float]] # [(timestamp, cx, cy), ...]
    raw_velocity: ImageVelocity
    smoothed_velocity: ImageVelocity
    is_active: bool = True
```

---

## 4. Centroid & Motion Estimation

### 4.1 Centroid Formulation
For any detected bounding box $(x_1, y_1, x_2, y_2)$:
$$c_x = \frac{x_1 + x_2}{2}, \quad c_y = \frac{y_1 + y_2}{2}$$

### 4.2 Image-Space Velocity Equations
Given previous centroid $(c_{x, prev}, c_{y, prev})$ at timestamp $t_{prev}$ and current centroid $(c_{x}, c_{y})$ at timestamp $t_{curr}$ with elapsed time $\Delta t = t_{curr} - t_{prev}$:

$$\Delta c_x = c_x - c_{x, prev}, \quad \Delta c_y = c_y - c_{y, prev}$$

$$v_x = \frac{\Delta c_x}{\Delta t}, \quad v_y = \frac{\Delta c_y}{\Delta t}$$

$$\text{speed} = \sqrt{v_x^2 + v_y^2}$$

### 4.3 Exponential Moving Average (EMA) Smoothing
To eliminate frame-to-frame bounding-box jitter without introducing excessive lag, an independent velocity EMA filter is applied ($\alpha = 0.20$):

$$v_{x, \text{smoothed}} = \alpha \cdot v_x + (1 - \alpha) \cdot v_{x, \text{prev\_smoothed}}$$
$$v_{y, \text{smoothed}} = \alpha \cdot v_y + (1 - \alpha) \cdot v_{y, \text{prev\_smoothed}}$$
$$\text{speed}_{\text{smoothed}} = \sqrt{v_{x, \text{smoothed}}^2 + v_{y, \text{smoothed}}^2}$$

---

## 5. Critical Research Distinction: Image Velocity vs. Physical Velocity

> [!IMPORTANT]
> **IMAGE VELOCITY $\neq$ PHYSICAL VELOCITY**
> - The computed velocity $(v_x, v_y)$ and speed are measured strictly in **pixels per second (px/s)** on the 2D image plane.
> - They do **NOT** represent real-world physical metric speed (m/s or km/h).
> - Factors such as perspective projection, camera focal length, optical zoom, distance from camera, and observer walking motion scale image velocity non-linearly.
> - Therefore, image velocity is solely used for spatial trajectory tracking and 2D bearing dynamics, **NOT** metric collision time.

---

## 6. Trajectory History & Track Lifecycle Management

### 6.1 Bounded Trajectory Buffer
- Handled by `TrajectoryHistory` using a FIFO circular buffer with configurable `max_history` (default: 30 points).
- Oldest observations are automatically evicted to prevent memory leaks over long walking sessions.

### 6.2 Track Eviction & Garbage Collection
- In the original baseline, `object_history` accumulated tracked keys infinitely, causing monotonic memory growth.
- In AVEN-VI, `ByteTrackLifecycleManager` tracks `missing_frames` for each track:
  - If a track is detected in frame $k$, `missing_frames = 0`.
  - If absent, `missing_frames += 1`.
  - When `missing_frames > max_missing_frames` (default: 15 frames / 0.5s), the track is evicted from active memory and its trajectory buffer is released.

---

## 7. Verification & Test Results

### 7.1 Automated Unit Tests (13/13 PASSED)
Executed via `pytest tests/ -v`:

| Test Module | Test Case | Condition Tested | Result |
| :--- | :--- | :--- | :---: |
| `test_centroid.py` | `test_centroid_standard` | Symmetric integer box (100, 100, 200, 200) | **PASS** |
| `test_centroid.py` | `test_centroid_asymmetric` | Asymmetric box (0, 10, 100, 50) | **PASS** |
| `test_centroid.py` | `test_centroid_floating_point` | Sub-pixel float box (10.5, 20.5, 30.5, 40.5) | **PASS** |
| `test_centroid.py` | `test_centroid_single_point` | Degenerate point box | **PASS** |
| `test_motion.py` | `test_missing_previous_observation` | Initial frame (dt > 0, no prior) -> zero velocity | **PASS** |
| `test_motion.py` | `test_zero_movement` | Stationary object -> zero velocity | **PASS** |
| `test_motion.py` | `test_horizontal_movement` | 100 px delta in 1.0 s -> vx=100 px/s, vy=0 | **PASS** |
| `test_motion.py` | `test_vertical_movement` | -50 px delta in 0.5 s -> vy=-100 px/s | **PASS** |
| `test_motion.py` | `test_diagonal_movement` | (30, 40) px delta in 1.0 s -> speed=50 px/s | **PASS** |
| `test_motion.py` | `test_ema_smoothing_step_response` | Step spike from 100 to 200 px/s dampened to 120 px/s | **PASS** |
| `test_trajectory.py` | `test_trajectory_history_limit` | FIFO buffer bounded at max_history=5, evicting oldest | **PASS** |
| `test_trajectory.py` | `test_trajectory_clear` | Buffer reset & cleanup | **PASS** |
| `test_trajectory.py` | `test_track_expiration_logic` | Missing frames > 3 evicted from active dict | **PASS** |

### 7.2 Integration Tests

1. **Test Run on `driving_video.mp4`** (90 frames, 1912x1076):
   - **Processed Frames**: 90 / 90
   - **Mean Latency**: **82.6 ms / frame**
   - **Throughput**: **10.6 FPS** (Pure CPU)
   - **Unique Tracks**: 4 vehicles detected and tracked with persistent IDs
   - **Output Saved**: `aven_vi_tracking_output.mp4`

2. **Test Run on `demo.mp4`** (315 frames, 1920x900):
   - **Processed Frames**: 315 / 315
   - **Mean Latency**: **61.1 ms / frame**
   - **Throughput**: **14.3 FPS** (Pure CPU)
   - **Unique Tracks**: 1 walking person detected, tracked with continuous trajectory trail
   - **Track Cleanup**: Active tracks at video conclusion = 0 (confirmed automatic eviction upon exit)
   - **Output Saved**: `aven_vi_demo_output.mp4`

---

## 8. Current Phase 2 Limitations

1. **Image Space Only**: Velocity is strictly 2D pixel displacement ($v_x, v_y$). Does not provide real-world metric speed or depth.
2. **Ego-Motion Coupling**: Observer camera motion (walking sway) adds a vector to all static scene objects. Ego-motion separation will be addressed in future phases.
3. **No Auditory Interface**: Output remains visual-only (for researcher debugging).
4. **No Collision or Proximity Assessment**: TTC and risk tiers remain deliberately omitted per Phase 2 instructions.
