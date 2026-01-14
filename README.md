# Tri-Modal Damage Detection for Aerial Inspection (RGB–Thermal–LiDAR)

This repository contains the ROS 2 Humble workspace used for a semester project on infrastructure crack inspection using a rigid multi-sensor rig:
- **RGB**: FLIR Blackfly S (Spinnaker driver)
- **Thermal**: Seek Thermal Mosaic Core
- **3D LiDAR**: Livox MID-360

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

## Workspace Setup (one terminal)
Open a new terminal and run:

```bash
cd ~/Time/ros2_ws
source /opt/ros/humble/setup.bash
source ~/ros2_humble/install/setup.bash
source install/setup.bash
