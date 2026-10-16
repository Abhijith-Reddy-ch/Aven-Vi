# 🚶 TTC-Enhanced Assistive Navigation System
### Predictive Collision Avoidance Using YOLOv8, MiDaS Depth Estimation & Grounded VQA

> **Shifting from Reactive Proximity Detection to Predictive Collision Avoidance**

An AI-powered assistive navigation system designed to help visually impaired individuals navigate their surroundings more safely. The system combines real-time object detection, depth estimation, object tracking, and Time-to-Collision (TTC) analysis to identify potential hazards and provide context-aware navigation alerts.

The proposed system focuses on **predicting collision risks rather than simply detecting nearby objects**, enabling more intelligent and dynamic hazard prioritization.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Proposed Solution](#-proposed-solution)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Methodology](#-methodology)
- [Technology Stack](#-technology-stack)
- [System Workflow](#-system-workflow)
- [Installation](#-installation)
- [Usage](#-usage)
- [Expected Output](#-expected-output)
- [Applications](#-applications)
- [Future Enhancements](#-future-enhancements)
- [Limitations](#-limitations)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

Traditional assistive navigation systems often rely on object proximity or fixed distance thresholds to generate alerts. However, an obstacle being nearby does not necessarily mean that it presents an immediate collision risk.

For example:
- A stationary wall may be close but not pose an immediate collision threat.
- A moving pedestrian may be farther away but approaching rapidly.
- Different environmental hazards may require different levels of urgency.

This project proposes a lightweight, CPU-oriented assistive navigation framework that combines visual perception, depth estimation, motion tracking, and collision-risk analysis.

The system integrates:

- **YOLOv8** for object detection.
- **MiDaS** for monocular depth estimation.
- **Centroid tracking** for tracking objects across video frames.
- **Time-to-Collision (TTC)** for estimating collision urgency.
- **Grounded Visual Question Answering (VQA)** for safety-related visual reasoning.
- **BLIP** for handling general visual question-answering queries.

The objective is to transform conventional obstacle detection into a more context-aware and predictive navigation system.

---

## ❗ Problem Statement

Conventional obstacle detection pipelines have several limitations:

### 1. Static Urgency
Objects may trigger identical warnings regardless of whether they are stationary or moving toward the user.

### 2. Pixel-Based Distance Estimation
Camera-specific bounding-box sizes and pixel-area heuristics do not reliably represent physical distance.

### 3. Lack of Motion Awareness
Without estimating closing velocity and Time-to-Collision, the system cannot adequately prioritize approaching hazards.

### 4. Disconnected Visual Reasoning
Visual question-answering systems may produce responses that do not align with navigation risk assessments.

### 5. Restricted Object Recognition
Systems designed around a single object category may fail to identify other important environmental hazards.

---

## 💡 Proposed Solution

The proposed framework introduces a unified perception-to-action pipeline.

Instead of relying exclusively on object proximity, the system analyzes the estimated depth and movement of detected objects to determine their potential collision urgency.

### Core Components

| Component | Responsibility |
|---|---|
| YOLOv8 | Detects objects in camera frames |
| MiDaS | Estimates relative depth from monocular images |
| Centroid Tracking | Maintains object positions across successive frames |
| Velocity Estimation | Estimates object movement and closing velocity |
| TTC Risk Engine | Estimates the time remaining before a potential collision |
| Decision Engine | Prioritizes hazards and determines alert urgency |
| Grounded VQA | Answers safety-related questions using the estimated risk state |
| BLIP | Supports general image-question-answering queries |
| Voice and Visual Output | Communicates navigation information and warnings |

---

## ✨ Key Features

- **Real-Time Object Detection:** Identifies environmental objects using YOLOv8.
- **Monocular Depth Estimation:** Uses MiDaS to estimate scene depth.
- **Object Tracking:** Tracks object centroids across successive video frames.
- **Motion-Aware Risk Assessment:** Estimates object movement and closing velocity.
- **Time-to-Collision Analysis:** Prioritizes hazards according to estimated collision urgency.
- **Dynamic Alert Prioritization:** Supports risk-based warning generation instead of relying exclusively on fixed proximity thresholds.
- **Visual Question Answering:** Integrates visual reasoning with navigation-related information.
- **CPU-Oriented Architecture:** Designed with deployment on standard edge-computing hardware in mind.
- **Zero-Retraining Approach:** Uses pretrained models rather than requiring a custom training phase for the proposed pipeline.
- **Extensible Hazard Recognition:** Aims to support recognition of hazards beyond a single object category.

---

## 🏗️ System Architecture

The system is organized into five major processing stages.

```text
                  ┌────────────────────────┐
                  │     Camera Input       │
                  │     RGB Video Frame     │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │    Perception Module    │
                  │                        │
                  │  YOLOv8 Object Detection│
                  │  MiDaS Depth Estimation│
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │ Spatial & Motion       │
                  │ Tracking               │
                  │                        │
                  │ Centroid Tracking      │
                  │ Velocity Estimation    │
                  │ Spatial Zone Analysis  │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │    TTC Risk Engine     │
                  │                        │
                  │ Collision-Time         │
                  │ Estimation             │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │    Decision Engine     │
                  │                        │
                  │ Risk Prioritization    │
                  │ TTC-Based Triggers     │
                  │ Alert Selection        │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │   Output & Reasoning   │
                  │                        │
                  │ Voice / Visual Alerts  │
                  │ Grounded VQA           │
                  │ General VQA (BLIP)     │
                  └────────────────────────┘
```

### Architecture Description

**1. Perception Module**

Receives camera frames and processes them using object detection and monocular depth estimation.

**2. Spatial and Motion Tracking**

Extracts object centroids, maintains tracking histories, estimates movement, and evaluates spatial zones.

**3. TTC Risk Engine**

Combines estimated depth and closing velocity to calculate collision urgency.

**4. Decision Engine**

Prioritizes detected hazards and determines the appropriate warning based on the estimated risk.

**5. Output and Reasoning**

Communicates relevant warnings and provides visual question-answering capabilities grounded in the current scene and risk state.

---

## 🧠 Methodology

### 1. Object Detection

YOLOv8 processes each video frame and identifies objects with their corresponding bounding boxes and class predictions.

The detections provide the spatial information required for subsequent depth estimation and tracking.

### 2. Depth Estimation

MiDaS estimates depth from monocular RGB images.

Depth information provides an additional spatial signal that can be used to reason about the relative positions of detected objects.

**Note:** Standard MiDaS outputs are generally relative-depth estimates, not inherently calibrated metric distances. Converting these predictions into reliable physical distances requires suitable calibration or an additional estimation method.

### 3. Centroid Tracking

For each detected object, the system calculates the bounding-box centroid.

For a bounding box represented by `(x1, y1, x2, y2)`:

```text
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

Centroids are associated across consecutive frames to maintain object histories and estimate movement.

### 4. Velocity Estimation

Object movement is estimated using changes in tracked positions or depth over time.

A simple velocity estimate is:

```text
Velocity = Change in Position / Elapsed Time
```

For depth-based collision prediction, the system must estimate the rate at which the distance between the user and an object is decreasing.

### 5. Time-to-Collision (TTC)

Time-to-Collision estimates how much time remains before an approaching object reaches a relevant collision distance.

A simplified TTC formulation is:

```text
TTC = Distance to Collision / Closing Velocity
```

For an object approaching the user:

```text
TTC = (Current Distance - Safe Distance) / Closing Speed
```

where:

- `Current Distance` is the estimated distance to the object.
- `Safe Distance` is the desired minimum separation.
- `Closing Speed` is the positive rate at which the separation decreases.

If the object is not approaching, a positive collision time may not be meaningful.

**Important:** TTC is a risk-estimation signal, not a guarantee that a collision will occur. Reliable calculations require consistent distance units, time measurements, and motion estimates.

### 6. Risk-Based Decision Making

The decision engine evaluates the available tracking and TTC information to prioritize potential hazards.

The intended behavior is to distinguish between:

- Nearby stationary obstacles.
- Approaching moving objects.
- Objects moving away from the user.
- Hazards with different estimated collision times.

Risk thresholds and alert policies should be calibrated and evaluated before deployment.

### 7. Grounded Visual Question Answering

The framework incorporates visual question answering to provide contextual information about the scene.

- **Grounded VQA:** Handles safety-related queries using the available spatial and risk information.
- **BLIP:** Routes general image-question-answering queries to a separate visual-language model.

The goal is to align natural-language responses with the system's navigation state rather than treating visual reasoning and hazard detection as independent processes.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| OpenCV | Camera access and image processing |
| YOLOv8 | Object detection |
| MiDaS | Monocular depth estimation |
| NumPy | Numerical computations |
| Centroid Tracking | Object association across frames |
| TTC-Based Risk Engine | Collision urgency estimation |
| Grounded VQA | Safety-related visual reasoning |
| BLIP | General visual question answering |
| Text-to-Speech | Optional spoken navigation alerts |

**Note:** The table describes the proposed technology stack. Confirm the actual dependencies against your implementation before finalizing the repository documentation.

---

## 📂 Suggested Project Structure

Organize the repository using a structure similar to the following:

```text
TTC-Enhanced-Assistive-Navigation/
│
├── models/
│   ├── yolov8/
│   ├── midas/
│   └── vqa/
│
├── src/
│   ├── detection.py
│   ├── depth_estimation.py
│   ├── tracking.py
│   ├── velocity_estimation.py
│   ├── ttc_engine.py
│   ├── risk_assessment.py
│   ├── decision_engine.py
│   ├── visual_qa.py
│   └── main.py
│
├── assets/
│   ├── architecture.png
│   └── demo.png
│
├── tests/
│
├── requirements.txt
├── README.md
└── LICENSE
```

This is a **suggested structure**, not a description of files that are already present in the repository.

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or a compatible version supported by your selected dependencies.
- A webcam or video source.
- A CPU-capable computer.
- Internet access for downloading pretrained model weights when required.

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd TTC-Enhanced-Assistive-Navigation
```

Replace the placeholder with your actual GitHub repository URL.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies specified by your implementation.

For example, the project may use:

```bash
pip install ultralytics opencv-python numpy torch torchvision transformers
```

Additional dependencies may be necessary depending on how MiDaS, BLIP, voice alerts, and Grounded VQA are implemented.

### Step 4: Download or Configure Model Weights

Configure the pretrained weights required by the object detection, depth estimation, and visual question-answering modules.

Ensure that model paths and configurations match the implementation.

---

## ▶️ Usage

The exact startup command depends on the entry point implemented in your repository.

If your application uses `src/main.py`, the following command may be appropriate:

```bash
python src/main.py
```

For webcam input, configure the camera index in the application.

For video-file input, configure the path to the video file.

Example configuration:

```python
CAMERA_SOURCE = 0
```

The value `0` typically represents the default webcam when using OpenCV.

### Expected Processing Sequence

1. Capture a frame from the camera.
2. Detect objects using YOLOv8.
3. Estimate scene depth using MiDaS.
4. Track detected objects across frames.
5. Estimate object movement and closing velocity.
6. Calculate TTC where the available estimates support it.
7. Determine the current hazard priority.
8. Generate the appropriate visual or voice alert.
9. Process safety-related questions using the grounded reasoning module.

**Important:** The commands above are examples based on the proposed architecture. Update them to match your actual implementation.

---

## 📊 Expected Output

Depending on the implemented modules, the system may display:

- Live camera frames.
- Object bounding boxes and class labels.
- Estimated depth information.
- Tracked object centroids.
- Estimated object velocity.
- TTC estimates.
- Risk categories or hazard priorities.
- Context-aware navigation alerts.
- Responses to supported visual questions.

A conceptual output could look like this:

```text
Detected Object : Pedestrian
Tracking Status : Active
Depth Estimate  : Available
Motion Status   : Approaching
TTC Estimate    : Computed when valid
Risk Assessment : Based on configured thresholds
Navigation Alert: Generated according to risk policy
```

This is an illustrative example, not a measured result from the system.

---

## 🌍 Applications

Potential applications include:

- Assistive navigation for visually impaired users.
- Pedestrian hazard awareness.
- Indoor obstacle detection.
- Smart mobility assistance.
- Context-aware visual navigation.
- Research into real-time collision prediction.
- Edge-computing applications involving visual perception.

The framework is intended to support navigation assistance, not replace established mobility aids or independent safety judgment.

---

## 🔮 Future Enhancements

Potential directions for further development include:

- Improved metric-depth calibration.
- More robust multi-object tracking.
- Trajectory prediction for moving hazards.
- Better handling of occlusion and missed detections.
- Adaptive TTC thresholds based on user movement.
- Spatially informative audio guidance.
- Support for additional open-vocabulary hazard classes.
- Performance optimization for low-power hardware.
- Quantitative evaluation using collision-prediction metrics.
- Testing under different lighting and environmental conditions.
- Integration with wearable cameras or mobile devices.

These are potential enhancements rather than claims about currently implemented features.

---

## ⚠️ Limitations

The system's reliability depends on the quality of object detection, depth estimation, and motion tracking.

Important limitations include:

- Monocular depth estimation may not provide accurate metric distances without calibration.
- Camera movement can affect apparent object motion.
- Occlusions and missed detections can interrupt tracking.
- TTC estimates become unreliable when velocity measurements are noisy or inconsistent.
- CPU-only execution may introduce latency depending on hardware and model configuration.
- Visual question-answering models may generate incorrect or incomplete responses.
- Unusual hazards may not be detected reliably.

Before real-world deployment, the system should undergo extensive testing for latency, false alarms, missed hazards, and collision-prediction accuracy.

**Safety Disclaimer:** This project is intended for research and prototyping. It should not be relied upon as the sole means of navigation or collision avoidance without appropriate validation and safety testing.

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Test the changes.
5. Submit a pull request with a clear description.

For significant changes, consider opening an issue before starting implementation.

---

## 📄 License

Choose a license appropriate for your project before distributing the code.

For example, you may use the MIT License by adding a `LICENSE` file containing its terms.

If the project incorporates pretrained models or third-party code, review their individual licenses and usage restrictions.

---

## 👨‍💻 Project Summary

**TTC-Enhanced Assistive Navigation** explores a shift from reactive obstacle detection to predictive collision avoidance by combining object detection, monocular depth estimation, centroid tracking, velocity estimation, and Time-to-Collision analysis.

By integrating visual perception with grounded language-based reasoning, the proposed framework aims to provide more context-aware and risk-sensitive navigation assistance.

### Core Technologies

`Python` · `YOLOv8` · `MiDaS` · `OpenCV` · `Centroid Tracking` · `Time-to-Collision` · `Grounded VQA` · `BLIP`

---
