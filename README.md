# Tri-Modal Damage Detection for Aerial Inspection (RGB–Thermal–LiDAR)

## Project Overview

This repository contains the codebase and ROS 2 workspace developed for a semester project on **infrastructure crack inspection** using a rigid multi-sensor platform. The original goal was to combine **RGB**, **thermal infrared**, and **LiDAR** sensing to (i) detect cracks robustly in challenging conditions and (ii) map detections into **3D** for metric analysis (e.g., location, extent, and geometry).

### Motivation
Crack inspection is traditionally performed manually and is time-consuming, subjective, and difficult to scale. Vision-based deep learning methods can detect cracks effectively in 2D, but they are sensitive to lighting changes, motion blur, shadows, and surface staining. Adding depth measurements enables **3D localization** and supports quantitative measurements that are not possible in image space alone.

### Sensor Rig
The system is built around a rigid sensor rig comprising:
- **RGB camera (FLIR Blackfly S)** for high-resolution texture and crack appearance
- **Thermal camera (Seek Mosaic Core)** to capture temperature patterns that may reveal damage cues under poor lighting (explored during the project)
- **3D LiDAR (Livox MID-360)** to provide dense 3D geometry and enable metric reasoning in a common reference frame

All sensors are interfaced through a Linux processing unit running **ROS 2 Humble**. RViz is used during acquisition for live point cloud visualization and basic validation.

### Calibration and Fusion Concept
To associate 2D detections with 3D geometry, the system requires:
- **Intrinsic calibration** of the RGB camera (performed offline)
- **Extrinsic calibration** between the LiDAR and camera frames to enable projection of LiDAR points into the image and back-projection of image regions into 3D

Several extrinsic calibration strategies were tested. Target-based calibration using **planar ArUco markers** provided the most reliable initial alignment, followed by a **qualitative alignment refinement** step to achieve visually consistent overlays.

### Crack Detection and 3D Reconstruction Pipeline
The intended processing pipeline is:
1. Detect crack regions in RGB using **YOLOv8**
2. Produce a pixel-level crack mask using **SAM2**
3. Use the calibrated LiDAR–camera geometry to associate the 2D crack mask with corresponding LiDAR points (projection / back-projection)
4. Extract crack-related 3D points and reconstruct a **3D mesh** for visualization and potential metrology

In practice, crack-to-3D association is challenging because cracks are thin structures and LiDAR returns may be dominated by background and surface roughness, especially at stand-off distances. Therefore, reported metrics should be interpreted as **indicative trends rather than conclusive results**, due to limited dataset size and operating conditions.

The pipeline supports:
1. Launching sensor drivers (RGB / Thermal / LiDAR)
2. Accumulating LiDAR scans into a denser point cloud (optional)
3. Recording synchronized RGB–thermal images with the corresponding LiDAR data (approximate time synchronization)
4. Live monitoring using **RViz** (point cloud visualization and sanity checks)

---

## Requirements
- Ubuntu + ROS 2 Humble
- Livox MID-360 driver installed and working
- Spinnaker SDK + `spinnaker_camera_driver` installed
- Seek thermal node built and runnable
- This workspace built with `colcon`

> The commands below assume the workspace path is: `~/Time/ros2_ws`

---
To install the requirements:
```bash
pip install -r requirements.txt
```

Open a new terminal and run:

```bash
cd ~/Time/ros2_ws
source /opt/ros/humble/setup.bash
source ~/ros2_humble/install/setup.bash
source install/setup.bash

```
Run the Livox driver

```bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```
Run the Livox accumulator

```bash
ros2 run flir_livox_sync livox_accumulator --ros-args -p window_duration:=20.0
```
Run the RGB driver

```bash
ros2 run spinnaker_camera_driver camera_driver_node \
  --ros-args \
  -r __node:=flir_camera \
  -p serial_number:="'25235670'" \
  -p parameter_file:="/home/semesterproject/Time/ros2_ws/install/spinnaker_camera_driver/share/spinnaker_camera_driver/config/blackfly_s.yaml"
  ```
  Run the Seek node
  
  ```bash
  ros2 run seek_thermal_cpp seek_thermal_node
  ```
  Run the synchronizer

  ```bash
  ros2 run flir_livox_sync flir_seek_livox_recorder \
  --ros-args -p slop:=0.5 -p output_root:=/home/semesterproject/ros2_data
  ```