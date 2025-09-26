""""
# CV-Distance-DEMO

A minimal demo for real-time distance estimation using a **single camera** and **OpenCV ArUco markers**.

## Features
- Capture live video from a USB camera
- Detect an ArUco square marker in real time
- Estimate camera pose using calibration parameters (`calib.npz`)
- Overlay measurement results directly on the video stream:
  - Green bounding box around the detected marker
  - Top-left corner:  
    - Line 1: overall median distance `D` (meters)  
    - Line 2: current FPS
  - Right side (blue column): relative position of the four corners (x%, y%) and each corner’s distance Z (meters)

## Quick Start

### 1. Install dependencies
Make sure you have OpenCV with the `aruco` module available:
```bash
pip install opencv-contrib-python numpy
```

### 2. Generate an ArUco marker
Run:
```bash
python generate_marker.py --dict DICT_5X5_50 --id 0 --out marker.png
```
Display the marker full-screen on your phone/tablet, or print it on paper.  
**Important:** measure the physical side length (in mm) with a ruler and update the constant `MARKER_MM` inside `distance_demo.py`.

### 3. Camera calibration
Use `capture_images.py` to capture multiple chessboard images, then calibrate the camera with your preferred tool to produce `calib.npz`.  
This file must include at least:
- `K` (camera intrinsic matrix)  
- `dist` (distortion coefficients)  
- `img_size` (image resolution used for calibration)

### 4. Run the demo
```bash
python distance_demo.py
```

If the marker is detected successfully, you will see real-time overlays with FPS, overall distance, and per-corner coordinates.
""""
