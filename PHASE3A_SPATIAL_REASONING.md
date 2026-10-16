# Phase 3A: Spatial Reasoning & Depth Integration Documentation

## Executive Overview
Phase 3A elevates **AVEN-VI** (Assistive Vision-Enabled Navigation for Visually Impaired Users) from 2D pixel tracking to a **3D spatial reasoning** system. It introduces:
1. **Clock-Face Directional Mapping** ($9:00 - 3:00$ navigation sectors)
2. **Central Walking Corridor Clearance Detection**
3. **Vertical Elevation Hazard Classification** (`GROUND`, `BODY`, `HEAD`)
4. **Calibrated Monocular Distance Approximation** ($0.3\text{m} - 25.0\text{m}$)
5. **Integrated Dense Monocular Depth Estimation** (Intel MiDaS Small) with mini depth radar visualization.

---

## 1. Mathematical Formulations & Spatial Algorithms

### 1.1 Clock-Face Directional Mapping
For wearable navigation, directions are translated into intuitive clock-face sectors:
- Camera Horizontal Field of View: $\text{HFOV} \approx 70^\circ$
- Normalized horizontal centroid: $u = c_x / W \in [0.0, 1.0]$
- Angular offset relative to optical axis:
  $$\theta = (u - 0.5) \cdot \text{HFOV}$$

| Bearing Angle Range ($\theta$) | Clock Position | Sector Meaning |
| :--- | :---: | :--- |
| $\theta < -25^\circ$ | **9:00** | Far Left Peripheral |
| $-25^\circ \le \theta < -15^\circ$ | **10:00** | Left Flank |
| $-15^\circ \le \theta < -5^\circ$ | **11:00** | Left Ahead (Approaching Path) |
| $-5^\circ \le \theta \le +5^\circ$ | **12:00** | **Direct Front (Immediate Heading)** |
| $+5^\circ < \theta \le +15^\circ$ | **1:00** | Right Ahead (Approaching Path) |
| $+15^\circ < \theta \le +25^\circ$ | **2:00** | Right Flank |
| $\theta > +25^\circ$ | **3:00** | Far Right Peripheral |

### 1.2 Central Walking Corridor Clearance
A pedestrian requires a clear forward corridor of width $W_{corridor}$ to step safely:
- Configurable corridor ratio $R_{corridor} = 0.30$ (central 30% of the image span)
- Horizontal corridor bounds:
  $$x_{min} = (0.5 - R_{corridor}/2) \cdot W, \quad x_{max} = (0.5 + R_{corridor}/2) \cdot W$$
- An obstacle is flagged **IN PATH** (`in_corridor = True`) if its horizontal bounding box $[x_1, x_2]$ intersects $[x_{min}, x_{max}]$.

### 1.3 Elevation Hazard Classification
Obstacles are categorized into three physical impact elevation zones:
- **GROUND HAZARD** ($y_1 \ge 0.50 \cdot H$ and $y_2 \ge 0.65 \cdot H$):
  Curbs, drop-offs, steps, low stools, pets, tripping objects.
- **BODY HAZARD** (spans mid-torso $y \in [0.25H, 0.85H]$):
  Direct collision threats: oncoming pedestrians, vehicles, bollards, doors.
- **HEAD HAZARD** ($y_1 \le 0.25 \cdot H$ and $y_2 < 0.65 \cdot H$):
  Overhanging threats: low tree branches, awnings, lifted truck gates, overhead signage.

### 1.4 Calibrated Distance Approximation
Monocular disparity is calibrated into approximate physical meters using weighted multi-cue fusion:
1. **Disparity Inversion**:
   $$d_{disp} = \frac{S_{scale}}{\text{disparity}_{median} + \delta}$$
2. **Ground-Plane Contact Geometry**:
   Using camera chest-mount height $h_{cam} \approx 1.35\text{ m}$ and vertical angle of lower box edge $y_2$:
   $$d_{ground} = \frac{h_{cam}}{\tan(\theta_{pitch} + (y_2/H - 0.5) \cdot \text{VFOV})}$$
3. **Fused Output**:
   $$d_{est} = 0.60 \cdot d_{disp} + 0.40 \cdot d_{ground} \in [0.3\text{m}, 25.0\text{m}]$$

> [!NOTE]
> **RESEARCH INTEGRITY**: Monocular depth provides calibrated metric approximations, not LiDAR-grade metrology. Scale variations can occur with unmodeled slopes or non-ground obstacles.

---

## 2. Modular Architecture (Phase 3A)

```text
aven_vi/
├── spatial/
│   ├── __init__.py
│   ├── clock_mapper.py         # Clock directions & walking corridor
│   ├── hazard_classifier.py    # Ground / Body / Head elevation zones
│   └── distance_calibrator.py  # Disparity & ground geometry fusion
├── perception/
│   ├── detector.py             # Multi-class YOLOv8
│   ├── tracker.py              # ByteTrack + Spatial reasoning integration
│   └── depth_estimator.py      # MiDaS Small dense depth & ROI extraction
├── motion/
│   ├── centroid.py
│   ├── trajectory.py
│   └── motion_estimator.py
├── visualization/
│   └── overlay.py              # HUD with clock, elevation, distance & corridor guides
└── utils/
    └── types.py                # TrackState updated with spatial fields
```

---

## 3. Unit Test Verification (20/20 PASSED)

Executed via `pytest tests/ -v`:
- `tests/test_clock_mapper.py`: Verified all 7 directional clock sectors ($9:00 - 3:00$) and corridor boundary overlaps (**PASS**)
- `tests/test_hazard_classifier.py`: Verified `GROUND`, `BODY`, and `HEAD` elevation categorizations (**PASS**)
- `tests/test_distance_calibrator.py`: Verified disparity scaling, ground contact geometry, and distance clamps (**PASS**)
- `tests/test_centroid.py`: 4 tests (**PASS**)
- `tests/test_motion.py`: 6 tests (**PASS**)
- `tests/test_trajectory.py`: 3 tests (**PASS**)

---

## 4. Integration Benchmarks

| Sequence | Frames | Mean Latency | Throughput | Unique Tracks | Spatial Output Video |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `driving_video.mp4` | 90 | **159.7 ms** | **5.5 FPS** | 4 vehicles | `aven_vi_spatial_output.mp4` (2.58 MB) |
| `demo.mp4` | 315 | **140.8 ms** | **6.3 FPS** | 1 person | `aven_vi_demo_spatial.mp4` (9.1 MB) |

Both runs produced full spatial visual overlays including:
- Clock-face bearing (e.g. `12:00 (+2°)`, `11:00 (-11°)`)
- Elevation tag (`[BODY]`, `[GROUND]`)
- Calibrated metric distance in meters (e.g. `2.4m`, `6.8m`)
- Yellow central pathway corridor guidelines with `[IN PATH]` hazard highlighting
- Live picture-in-picture Inferno depth radar.
