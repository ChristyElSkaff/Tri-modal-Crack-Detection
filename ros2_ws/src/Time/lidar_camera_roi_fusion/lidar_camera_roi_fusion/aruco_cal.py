#!/usr/bin/env python3
import json
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge

from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs_py import point_cloud2


def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


def ransac_plane(points, iters=250, thresh=0.02, min_inliers=800):
    # plane: n·p + d = 0
    N = points.shape[0]
    if N < 200:
        return None
    rng = np.random.default_rng()
    best = None
    best_in = 0

    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        p1, p2, p3 = points[idx]
        n = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -float(np.dot(n, p1))
        dist = np.abs(points @ n + d)
        mask = dist < thresh
        k = int(np.sum(mask))
        if k > best_in:
            best_in = k
            best = (n, d, mask)

    if best is None or best_in < min_inliers:
        return None

    n, d, mask = best

    # refine via SVD on inliers
    P = points[mask]
    c = np.mean(P, axis=0)
    X = P - c
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = normalize(vh[-1])
    d = -float(np.dot(n, c))
    dist = np.abs(points @ n + d)
    mask = dist < thresh
    return n, d, mask


def plane_basis(n):
    n = normalize(n)
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = normalize(np.cross(n, a))
    e2 = normalize(np.cross(n, e1))
    return e1, e2


def umeyama(A, B):
    # Find R,t s.t. B ≈ R A + t  (A,B are Nx3)
    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    XA = A - muA
    XB = B - muB
    H = XA.T @ XB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = muB - R @ muA
    return R, t


def rot_to_quat(Rm):
    t = np.trace(Rm)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        qw = 0.25 * S
        qx = (Rm[2,1] - Rm[1,2]) / S
        qy = (Rm[0,2] - Rm[2,0]) / S
        qz = (Rm[1,0] - Rm[0,1]) / S
    else:
        if (Rm[0,0] > Rm[1,1]) and (Rm[0,0] > Rm[2,2]):
            S = np.sqrt(1.0 + Rm[0,0] - Rm[1,1] - Rm[2,2]) * 2
            qw = (Rm[2,1] - Rm[1,2]) / S
            qx = 0.25 * S
            qy = (Rm[0,1] + Rm[1,0]) / S
            qz = (Rm[0,2] + Rm[2,0]) / S
        elif Rm[1,1] > Rm[2,2]:
            S = np.sqrt(1.0 + Rm[1,1] - Rm[0,0] - Rm[2,2]) * 2
            qw = (Rm[0,2] - Rm[2,0]) / S
            qx = (Rm[0,1] + Rm[1,0]) / S
            qy = 0.25 * S
            qz = (Rm[1,2] + Rm[2,1]) / S
        else:
            S = np.sqrt(1.0 + Rm[2,2] - Rm[0,0] - Rm[1,1]) * 2
            qw = (Rm[1,0] - Rm[0,1]) / S
            qx = (Rm[0,2] + Rm[2,0]) / S
            qy = (Rm[1,2] + Rm[2,1]) / S
            qz = 0.25 * S
    q = np.array([qx, qy, qz, qw], dtype=float)
    return q / np.linalg.norm(q)


class ArucoPlaneCalibrate(Node):
    def __init__(self):
        super().__init__("aruco_plane_calibrate")
        self.bridge = CvBridge()

        # params
        self.declare_parameter("image_topic", "/flir_camera/image_raw")
        self.declare_parameter("camera_info_topic", "/flir_camera/camera_info")
        self.declare_parameter("cloud_topic", "/livox/scan_window")

        self.declare_parameter("aruco_dict", "DICT_4X4_50")
        self.declare_parameter("aruco_id", 0)
        self.declare_parameter("tag_size_m", 0.12)

        self.declare_parameter("board_w_m", 1.0)
        self.declare_parameter("board_h_m", 0.7)
        self.declare_parameter("tag_center_dx_m", 0.0)  # tag center offset from board center (x right)
        self.declare_parameter("tag_center_dy_m", 0.0)  # (y down) in board plane coords

        self.declare_parameter("ransac_thresh_m", 0.02)
        self.declare_parameter("ransac_min_inliers", 800)

        self.declare_parameter("save_path", "/home/semesterproject/Time/extrinsics_samples.json")
        self.declare_parameter("need_samples", 15)

        self.K = None
        self.D = None
        self.w = None
        self.h = None

        self.samples = []

        self.create_subscription(CameraInfo, self.get_parameter("camera_info_topic").value, self.on_caminfo, 10)

        self.sub_img = Subscriber(self, Image, self.get_parameter("image_topic").value)
        self.sub_cloud = Subscriber(self, PointCloud2, self.get_parameter("cloud_topic").value)
        self.sync = ApproximateTimeSynchronizer([self.sub_img, self.sub_cloud], queue_size=10, slop=0.3)
        self.sync.registerCallback(self.on_sync)

        self.get_logger().info("Aruco+Plane calibration node started.")

    def on_caminfo(self, msg: CameraInfo):
        if self.K is None:
            self.w, self.h = msg.width, msg.height
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64) if len(msg.d) else None
            self.get_logger().info(f"CameraInfo loaded: {self.w}x{self.h}")

    def on_sync(self, img_msg: Image, cloud_msg: PointCloud2):
        if self.K is None:
            return

        # --- Detect ArUco ---
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dict_name = self.get_parameter("aruco_dict").value
        aruco_id = int(self.get_parameter("aruco_id").value)
        tag_size = float(self.get_parameter("tag_size_m").value)

        aruco = cv2.aruco
        aruco_dict = getattr(aruco, dict_name)
        dictionary = aruco.getPredefinedDictionary(aruco_dict)
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, params)

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or aruco_id not in ids.flatten().tolist():
            return

        idx = np.where(ids.flatten() == aruco_id)[0][0]
        c = corners[idx].reshape(4, 2)

        # object points for tag corners in tag frame (z=0)
        s = tag_size / 2.0
        obj = np.array([[-s, -s, 0],
                        [ s, -s, 0],
                        [ s,  s, 0],
                        [-s,  s, 0]], dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(obj, c, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return

        R_cam_tag, _ = cv2.Rodrigues(rvec)
        t_cam_tag = tvec.reshape(3)

        # --- Compute board corners in camera frame ---
        W = float(self.get_parameter("board_w_m").value)
        H = float(self.get_parameter("board_h_m").value)
        dx = float(self.get_parameter("tag_center_dx_m").value)
        dy = float(self.get_parameter("tag_center_dy_m").value)

        # board frame: origin at board center, x right, y down, z out of board
        # tag is assumed aligned with board (no rotation) and sitting on board plane z=0
        # tag center relative to board center = (dx, dy, 0) in board coords
        # board corners in board coords:
        board_corners_board = np.array([
            [-W/2, -H/2, 0],
            [ W/2, -H/2, 0],
            [ W/2,  H/2, 0],
            [-W/2,  H/2, 0]
        ], dtype=np.float64)

        # transform board->tag (since tag pose in camera is known)
        # If tag is at (dx,dy) in board coords, then:
        # p_tag = p_board - [dx,dy,0]
        board_to_tag_t = np.array([dx, dy, 0.0], dtype=np.float64)

        board_corners_tag = board_corners_board - board_to_tag_t

        # p_cam = R_cam_tag * p_tag + t_cam_tag
        board_corners_cam = (R_cam_tag @ board_corners_tag.T).T + t_cam_tag

        # --- LiDAR: find board plane and rectangle corners ---
        pts = []
        for p in point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        if len(pts) < 500:
            return
        pts = np.array(pts, dtype=np.float64)

        # fit a plane (assumes board is dominant plane in front; keep scene simple)
        thresh = float(self.get_parameter("ransac_thresh_m").value)
        min_in = int(self.get_parameter("ransac_min_inliers").value)
        model = ransac_plane(pts, iters=250, thresh=thresh, min_inliers=min_in)
        if model is None:
            return
        n, d, mask = model
        P = pts[mask]

        # project plane points to 2D coordinates
        e1, e2 = plane_basis(n)
        uv = np.stack([P @ e1, P @ e2], axis=1).astype(np.float32)

        # min area rectangle in 2D
        rect = cv2.minAreaRect(uv)
        box2 = cv2.boxPoints(rect)  # 4x2
        box2 = box2.astype(np.float64)

        # back to 3D in LiDAR frame: p = u*e1 + v*e2 + p0 (choose p0 on plane)
        # find a point on plane: n·p + d = 0 => p0 = -d*n (works since ||n||=1)
        p0 = -d * n
        corners_lidar = np.array([p0 + box2[i,0]*e1 + box2[i,1]*e2 for i in range(4)], dtype=np.float64)

        # --- Solve LiDAR->Camera using permutation search (because corner order may differ) ---
        best = None
        best_err = 1e9

        import itertools
        for perm in itertools.permutations(range(4)):
            A = corners_lidar[np.array(perm)]       # Nx3 in LiDAR
            B = board_corners_cam                  # Nx3 in camera
            R, t = umeyama(A, B)
            A2 = (R @ A.T).T + t
            err = np.sqrt(np.mean(np.sum((A2 - B)**2, axis=1)))
            if err < best_err:
                best_err = err
                best = (R, t)

        if best is None:
            return

        R, t = best
        q = rot_to_quat(R)

        self.samples.append({
            "t": t.tolist(),
            "q": q.tolist(),
            "rmse_m": float(best_err),
            "cloud_frame": cloud_msg.header.frame_id
        })

        self.get_logger().info(f"sample {len(self.samples)} rmse={best_err:.4f} m")

        need = int(self.get_parameter("need_samples").value)
        if len(self.samples) >= need:
            # robust aggregate: median translation, average quaternion
            T = np.array([s["t"] for s in self.samples], dtype=np.float64)
            Q = np.array([s["q"] for s in self.samples], dtype=np.float64)

            t_med = np.median(T, axis=0)

            # quaternion average (simple + works): normalize sum (ensure same hemisphere)
            Q2 = Q.copy()
            for i in range(1, Q2.shape[0]):
                if np.dot(Q2[0], Q2[i]) < 0:
                    Q2[i] *= -1
            q_avg = normalize(np.sum(Q2, axis=0))

            out = {
                "cloud_frame": self.samples[-1]["cloud_frame"],
                "t_med": t_med.tolist(),
                "q_avg": q_avg.tolist(),
                "samples": self.samples
            }

            path = self.get_parameter("save_path").value
            with open(path, "w") as f:
                json.dump(out, f, indent=2)

            self.get_logger().info("=== FINAL (LiDAR -> camera_optical) ===")
            self.get_logger().info(f"t = {t_med.tolist()}")
            self.get_logger().info(f"q = {q_avg.tolist()}")
            self.get_logger().info(f"Saved to {path}")
            self.get_logger().info("You can stop the node now.")

            # stop collecting more
            self.samples = []


def main():
    rclpy.init()
    node = ArucoPlaneCalibrate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()