# Touchless Ergonomic Posture Monitor

A real-time posture monitoring system that uses computer vision and deep learning to detect poor ergonomic habits - specifically slouching and sitting too close to the screen - and alerts the user via desktop notifications.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [How It Works](#how-it-works)
4. [Technology Stack](#technology-stack)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)
7. [Usage](#usage)
8. [Algorithm & Methodology](#algorithm--methodology)
9. [Calibration System](#calibration-system)
10. [Alert Mechanism](#alert-mechanism)
11. [Key Design Decisions](#key-design-decisions)
12. [Limitations](#limitations)
13. [Future Enhancements](#future-enhancements)
14. [References](#references)

---

## Overview

The **Touchless Ergonomic Posture Monitor** is a desktop application that leverages the YOLOv8 pose estimation model to monitor a user's sitting posture in real time through a webcam. The system detects two primary ergonomic issues:

- **Slouching** - when the user's spine angle deviates significantly from their calibrated upright posture.
- **Sitting too close** - when the user's face (measured by inter-ear distance in pixels) exceeds a threshold relative to their calibrated baseline.

When either condition persists for more than 5 seconds, the system fires a desktop notification reminding the user to correct their posture.

---

## Features

- **Real-time pose estimation** using YOLOv8 medium pose model running at webcam frame rates.
- **Automatic calibration** - the system learns the user's good posture over a 5-second window upon startup.
- **Dual detection** - monitors both spinal angle (slouching) and face-to-screen distance (proximity).
- **Temporal smoothing** - uses a rolling average over the last 8 frames to reduce false positives from momentary detection noise.
- **Desktop notifications** - uses the `plyer` library to send native OS notifications with a cooldown period to avoid spam.
- **Manual recalibration** - press `c` at any time to re-run the calibration routine.
- **Visual feedback** - live overlay showing detected keypoints, posture metrics, thresholds, and alert status.

---

## How It Works

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Webcam     │────▶│  YOLOv8 Pose     │────▶│  Keypoint       │
│   Feed       │     │  Estimation      │     │  Extraction     │
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Desktop     │◀────│  Alert Logic     │◀────│  Angle &        │
│  Notification│     │  (5s threshold)  │     │  Distance Calc  │
└──────────────┘     └──────────────────┘     └─────────────────┘
                                ▲
                                │
                       ┌────────┴────────┐
                       │  Auto-Calibrator │
                       │  (5s baseline)   │
                       └─────────────────┘
```

1. The webcam captures frames and passes them to the YOLOv8 pose model.
2. The model returns 17 COCO-format keypoints per detected person.
3. The system extracts ear keypoints (indices 3, 4) and shoulder keypoints (indices 5, 6).
4. From these, it computes:
   - **Inter-ear distance** (pixels) - a proxy for how close the user is to the camera.
   - **Posture angle** - the angle between the vertical axis and the line connecting mid-ear to mid-shoulder.
5. During calibration, baseline values are established. After calibration, deviations trigger alerts.

---

## Technology Stack

| Component          | Technology                          |
|--------------------|-------------------------------------|
| Language           | Python 3.10+                        |
| Deep Learning      | YOLOv8 (Ultralytics)                |
| Pose Model         | `yolov8m-pose.pt` (medium, 26.9 MB) |
| Computer Vision    | OpenCV (`cv2`)                      |
| Numerical Compute  | NumPy                               |
| Notifications      | Plyer                               |
| Video Capture      | OpenCV `VideoCapture`               |

---

## Project Structure

```
DL Project/
├── main.py                  # Main application entry point
├── yolov8m-pose.pt          # Pre-trained YOLOv8 medium pose weights
├── requirements.txt         # Python dependencies
├── Readme.md                # This file
└── main.tex/                # Project Report Code
└── llncs.cls/               # Springer Format
└── venv/                    # Virtual environment
```

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher (preferably 3.11)
- A webcam (built-in or external)
- Windows, macOS, or Linux

### Steps

1. **Clone or download** the project files.

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the model file** `yolov8m-pose.pt` is in the same directory as `main.py`.

5. **Run the application**:

   ```bash
   python main.py
   ```

---

## Usage

1. Run `python main.py`.
2. A window titled **"Touchless Ergonomic Monitor"** will open showing your webcam feed.
3. **Sit upright** in your normal working posture during the 5-second calibration period.
4. After calibration, the system begins monitoring:
   - If you **slouch** or **lean too close** for more than 5 seconds, a desktop notification appears.
5. **Controls:**
   - Press `c` - Recalibrate (re-run the 5-second baseline capture).
   - Press `q` - Quit the application.

---

## Algorithm & Methodology

### Keypoint Selection

The YOLOv8 pose model outputs 17 keypoints in COCO format. This project uses four:

| Keypoint Index | Body Part     |
|----------------|---------------|
| 3              | Left Ear      |
| 4              | Right Ear     |
| 5              | Left Shoulder |
| 6              | Right Shoulder |

### Distance Calculation (Proximity Detection)

The inter-ear distance serves as a proxy for face-to-camera distance:

```
ear_distance = sqrt((r_ear_x - l_ear_x)² + (r_ear_y - l_ear_y)²)
```

As the user moves closer to the camera, the ears appear farther apart in pixel space. A rolling mean over 8 frames smooths this value.

### Angle Calculation (Slouch Detection)

The posture angle is computed using the `calculate_angle` function with three points:

- **Point A**: A reference vertical point directly above the mid-shoulder: `(mid_shoulder_x, 0)`
- **Point B**: The mid-shoulder point
- **Point C**: The mid-ear point

```
angle = |arctan2(C.y - B.y, C.x - B.x) - arctan2(A.y - B.y, A.x - B.x)| × (180/π)
```

When the user sits upright, this angle is close to 0°. As they slouch forward, the angle increases. The system alerts when the angle drops below the calibrated threshold (indicating forward lean).

### Temporal Smoothing

Both ear distance and posture angle are stored in a `deque` with `maxlen=8`. The displayed and evaluated values are the **mean** of the last 8 frames, which filters out single-frame detection jitter.

---

## Calibration System

The `AutoCalibrator` class implements a two-phase approach:

### Phase 1: Sampling (5 seconds)

- The user sits in their correct working posture.
- Every frame's ear distance and posture angle are recorded.
- A countdown is displayed: `"CALIBRATING... Sit straight! (Xs)"`.

### Phase 2: Threshold Computation

After 5 seconds, two thresholds are computed:

| Threshold            | Formula                          | Meaning                          |
|----------------------|----------------------------------|----------------------------------|
| `too_close_thresh`   | `base_ear_dist × 1.20`           | 20% closer than baseline         |
| `slouch_thresh`      | `base_angle − 12.0`             | 12° more slouched than baseline |

These personalized thresholds account for different body types, camera distances, and seating setups.

---

## Alert Mechanism

The alert logic follows a **duration-based trigger** to avoid nuisance notifications:

1. If slouching OR too-close is detected, a timer starts.
2. If the condition persists for **≥ 5 seconds**, a notification is sent.
3. A **10-second cooldown** prevents notification spam.
4. If the user corrects their posture at any point, the timer resets.

### Notification Example

```
Title:   Ergonomic Alert
Message: You have been slouching for over 5.0 seconds! Sit up straight.
App:     Posture Monitor
Timeout: 4 seconds
```

---

## Key Design Decisions

| Decision                        | Rationale                                                  |
|---------------------------------|------------------------------------------------------------|
| YOLOv8 medium pose              | Best trade-off between accuracy (~68.6% AP) and speed      |
| Rolling average (8 frames)      | Eliminates false positives from detection jitter           |
| Auto-calibration                | Personalizes thresholds for each user and environment      |
| Duration-based alert trigger    | Prevents alerts for momentary posture lapses               |
| Notification cooldown (10s)     | Avoids overwhelming the user with repeated alerts          |
| Inter-ear distance as proxy     | Simple, effective proxy for face-to-screen distance        |
| COCO keypoints 3–4, 5–6         | Ears and shoulders are the most stable upper-body landmarks |

---

## Limitations

- **Single-person only** - the system processes the first detected person.
- **Frontal assumption** - works best when the user faces the camera; extreme side angles reduce accuracy.
- **Lighting dependent** - poor lighting can degrade pose estimation quality.
- **No seat detection** - cannot detect if the user has stood up and left.
- **2D analysis** - depth estimation is approximate (pixel-based, not metric).
- **Upper body only** - does not monitor leg position, armrest height, or monitor height.

---

## Future Enhancements

- **Multi-person support** - track and alert multiple users in frame.
- **Session analytics** - log posture data over time and generate daily/weekly reports.
- **Audio alerts** - optional sound notifications in addition to desktop popups.
- **Mobile companion app** - view posture reports on a phone.
- **Break reminders** - periodic reminders to stand up and stretch (Pomodoro-style).
- **Improved depth estimation** - use a depth camera or stereo vision for metric distance.
- **Head tilt detection** - monitor neck angle for forward head posture ("tech neck").
- **GUI settings panel** - allow users to adjust sensitivity, calibration time, and thresholds.

---

## References

- Ultralytics YOLOv8: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- COCO Keypoint Format: [https://cocodataset.org/#keypoints-2020](https://cocodataset.org/#keypoints-2020)
- OpenCV Documentation: [https://docs.opencv.org/](https://docs.opencv.org/)
- Plyer (Platform-independent notifications): [https://plyer.readthedocs.io/](https://plyer.readthedocs.io/)
