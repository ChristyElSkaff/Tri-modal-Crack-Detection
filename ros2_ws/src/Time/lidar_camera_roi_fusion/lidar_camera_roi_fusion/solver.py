#!/usr/bin/env python3
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# === CHANGE THIS if needed ===
PAIRS_PATH = "/home/semesterproject/Time/pairs_lidar_cam.json"

# Paste your K here (from your CameraInfo)
K = np.array([
    [3201.976442071439, 0.0, 807.1554510830895],
    [0.0, 3202.14, 483.66],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# If you want, you can later include distortion, but start with None for robustness
DIST = None

def main():
    pairs = json.load(open(PAIRS_PATH, "r"))
    if len(pairs) < 6:
        raise RuntimeError("Need at least 6 pairs, preferably 10+")

    frame_ids = {p["frame_id"] for p in pairs}
    print("Frame IDs in pairs:", frame_ids)

    obj = np.array([[p["X"], p["Y"], p["Z"]] for p in pairs], dtype=np.float64)  # LiDAR frame
    img = np.array([[p["u"], p["v"]] for p in pairs], dtype=np.float64)          # pixels

    ok, rvec, tvec = cv2.solvePnP(obj, img, K, DIST, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")

    R_cam_lidar, _ = cv2.Rodrigues(rvec)  # maps LiDAR->Cam: X_cam = R * X_lidar + t
    t = tvec.reshape(3)

    # reprojection error
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, DIST)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img, axis=1)
    print("Reprojection error px: mean=", float(np.mean(err)), "max=", float(np.max(err)))

    # quaternion
    quat = R.from_matrix(R_cam_lidar).as_quat()  # x,y,z,w
    print("\n=== LiDAR_FRAME -> CAMERA_OPTICAL transform ===")
    print("t (x y z):", t.tolist())
    print("q (x y z w):", quat.tolist())

    print("\nros2 static TF command (edit parent frame!):")
    print("ros2 run tf2_ros static_transform_publisher \\")
    print(f"  {t[0]} {t[1]} {t[2]} \\")
    print(f"  {quat[0]} {quat[1]} {quat[2]} {quat[3]} \\")
    print("  <LIDAR_FRAME> flir_camera_optical_frame")

if __name__ == "__main__":
    main()