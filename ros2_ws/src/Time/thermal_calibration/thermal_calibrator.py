#!/usr/bin/env python3
import os
from typing import List

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import yaml


class ThermalCalibrator(Node):
    def __init__(self):
        super().__init__('thermal_calibrator')

        # === Parameters ===
        self.declare_parameter('image_topic', '/thermal/image_raw')
        self.declare_parameter('output_yaml', 'thermal_intrinsics.yaml')
        # Your circles have ~3 cm spacing between centers -> 30 mm
        self.declare_parameter('square_size', 30.0)
        self.declare_parameter('min_samples', 15)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.output_yaml = self.get_parameter('output_yaml').get_parameter_value().string_value
        self.square_size = self.get_parameter('square_size').get_parameter_value().double_value
        self.min_samples = self.get_parameter('min_samples').get_parameter_value().integer_value

        # Asymmetric circle grid from acircles_pattern.pdf:
        # 4 circles per row, 11 rows (total 44 circles)
        self.pattern_size = (4, 11)  # (cols, rows)

        self.bridge = CvBridge()

        self.objpoints: List[np.ndarray] = []
        self.imgpoints: List[np.ndarray] = []
        self.image_shape = None

        # 3D coordinates of the pattern points
        self.objp = self.build_asymmetric_circle_grid(self.pattern_size, self.square_size)

        # Blob detector
        self.blob_detector = self.create_blob_detector()

        # Subscriber
        self.sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        # Services
        self.calib_srv = self.create_service(
            Trigger,
            'run_calibration',
            self.handle_run_calibration
        )
        self.clear_srv = self.create_service(
            Trigger,
            'clear_samples',
            self.handle_clear_samples
        )

        self.get_logger().info(f"ThermalCalibrator listening on {image_topic}")
        self.get_logger().info("Move the circle-board in front of the camera to collect samples.")
        self.get_logger().info(
            "Call 'ros2 service call /run_calibration std_srvs/srv/Trigger {}' when ready."
        )

    # ----------------------------------------------------------
    # Build 3D object points for asymmetric circle grid 4x11
    # Pattern from OpenCV example:
    # x = (2*j + i%2) * square_size
    # y = i * square_size
    # ----------------------------------------------------------
    def build_asymmetric_circle_grid(self, pattern_size, square_size):
        cols, rows = pattern_size  # cols = 4, rows = 11
        objp = np.zeros((rows * cols, 3), np.float32)
        idx = 0
        for i in range(rows):
            for j in range(cols):
                objp[idx, 0] = (2 * j + (i % 2)) * square_size
                objp[idx, 1] = i * square_size
                idx += 1
        return objp

    # ----------------------------------------------------------
    # Blob detector: small-ish round-ish blobs, color-agnostic
    # ----------------------------------------------------------
    def create_blob_detector(self):
        params = cv.SimpleBlobDetector_Params()

        # Don't filter by color: accept both dark and bright blobs
        params.filterByColor = False

        # Area tuned for your circle size
        params.filterByArea = True
        params.minArea = 30      # cut tiny noise
        params.maxArea = 400     # increase if your circles are much bigger

        # Circularity: favor round blobs
        params.filterByCircularity = True
        params.minCircularity = 0.5

        # Inertia & convexity: moderately strict
        params.filterByInertia = True
        params.minInertiaRatio = 0.3

        params.filterByConvexity = True
        params.minConvexity = 0.7

        return cv.SimpleBlobDetector_create(params)

    # ----------------------------------------------------------
    # Image callback: robust detection + precise refinement
    # ----------------------------------------------------------
    def image_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")
            return

        # --- Build two versions of the image ---
        # gray_raw: normalized but not blurred (for refinement & visualization)
        # gray_proc: inverted + blurred (for robust detection)
        if cv_img.ndim == 3:
            gray_raw = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
        else:
            if cv_img.dtype != np.uint8:
                gray_raw = cv.normalize(cv_img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
            else:
                gray_raw = cv_img

        if self.image_shape is None:
            self.image_shape = (gray_raw.shape[1], gray_raw.shape[0])  # (width, height)

        # Detection image: invert + light blur
        gray_proc = cv.bitwise_not(gray_raw)
        gray_proc = cv.medianBlur(gray_proc, 3)  # lighter blur than before

        # Asymmetric grid + clustering
        flags = cv.CALIB_CB_ASYMMETRIC_GRID | cv.CALIB_CB_CLUSTERING

        ret, centers = cv.findCirclesGrid(
            gray_proc,
            self.pattern_size,
            flags=flags,
            blobDetector=self.blob_detector
        )

        if ret:
            # Refine centers on the cleaner raw image with a smaller window
            gray_float = gray_raw.astype(np.float32)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            centers2 = cv.cornerSubPix(
                gray_float,
                centers,
                (5, 5),   # tighter refinement window
                (-1, -1),
                criteria
            )

            self.objpoints.append(self.objp.copy())
            self.imgpoints.append(centers2)

            self.get_logger().info(
                f"Pattern detected and stored. Total samples: {len(self.objpoints)}"
            )

            # Visualize on the raw (non-inverted) image
            vis = cv.cvtColor(gray_raw, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(vis, self.pattern_size, centers2, ret)
            cv.imshow('thermal_calibration', vis)
            cv.waitKey(1)
        else:
            # Debug: show blobs used for detection on the processed image
            vis = cv.cvtColor(gray_proc, cv.COLOR_GRAY2BGR)
            keypoints = self.blob_detector.detect(gray_proc)
            cv.drawKeypoints(
                gray_proc,
                keypoints,
                vis,
                flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            cv.imshow('thermal_calibration', vis)
            cv.waitKey(1)

    # ----------------------------------------------------------
    # Service: run calibration with per-view error filtering
    # ----------------------------------------------------------
    def handle_run_calibration(self, request, response):
        total = len(self.objpoints)
        if total < self.min_samples:
            response.success = False
            response.message = (
                f"Not enough samples: {total} collected, "
                f"{self.min_samples} required."
            )
            self.get_logger().warn(response.message)
            return response

        if self.image_shape is None:
            response.success = False
            response.message = "No images seen yet; cannot calibrate."
            self.get_logger().warn(response.message)
            return response

        # Limit number of views used (for speed and stability)
        max_views = 40
        if total > max_views:
            self.get_logger().info(
                f"Using only the first {max_views} of {total} collected samples for initial calibration."
            )
            objpoints = self.objpoints[:max_views]
            imgpoints = self.imgpoints[:max_views]
        else:
            objpoints = self.objpoints
            imgpoints = self.imgpoints

        self.get_logger().info(
            f"Initial cv.calibrateCamera with {len(objpoints)} views..."
        )

        # ---- Initial calibration ----
        ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints,
            imgpoints,
            self.image_shape,
            None,
            None
        )

        self.get_logger().info(f"Initial RMS reprojection error: {ret}")

        # ---- Compute per-view reprojection errors ----
        per_view_errors = []
        for i, (obj, img) in enumerate(zip(objpoints, imgpoints)):
            proj, _ = cv.projectPoints(obj, rvecs[i], tvecs[i], K, dist)
            proj = proj.reshape(-1, 2)
            img2 = img.reshape(-1, 2)
            err = np.sqrt(np.mean(np.sum((proj - img2) ** 2, axis=1)))
            per_view_errors.append(err)

        per_view_errors = np.array(per_view_errors)
        mean_err = float(np.mean(per_view_errors))
        std_err = float(np.std(per_view_errors))
        self.get_logger().info(f"Per-view errors (px): {per_view_errors.tolist()}")
        self.get_logger().info(f"Per-view mean: {mean_err:.3f}, std: {std_err:.3f}")

        # ---- Drop outlier views with very high error ----
        threshold = mean_err + 1.5 * std_err
        keep_indices = [i for i, e in enumerate(per_view_errors) if e <= threshold]

        self.get_logger().info(
            f"Outlier threshold: {threshold:.3f} px, keeping {len(keep_indices)} / {len(objpoints)} views."
        )

        if len(keep_indices) >= self.min_samples and len(keep_indices) < len(objpoints):
            # Filter views
            objpoints_f = [objpoints[i] for i in keep_indices]
            imgpoints_f = [imgpoints[i] for i in keep_indices]

            self.get_logger().info(
                f"Refining calibration with {len(objpoints_f)} filtered views..."
            )

            ret2, K2, dist2, rvecs2, tvecs2 = cv.calibrateCamera(
                objpoints_f,
                imgpoints_f,
                self.image_shape,
                None,
                None
            )

            self.get_logger().info(f"Refined RMS reprojection error: {ret2}")

            # Use refined calibration if it improved things
            if ret2 < ret:
                self.get_logger().info(
                    "Refined calibration improved RMS; using refined parameters."
                )
                K, dist, ret = K2, dist2, ret2
            else:
                self.get_logger().info(
                    "Refined calibration did not improve RMS; keeping initial parameters."
                )
        else:
            self.get_logger().info(
                "Not enough views after outlier filtering or no outliers detected; "
                "keeping initial calibration."
            )

        self.get_logger().info(f"Final RMS reprojection error: {ret}")
        self.get_logger().info(f"Final camera matrix K:\n{K}")
        self.get_logger().info(f"Final distortion coefficients:\n{dist.ravel()}")

        # ---- Save to YAML ----
        calib_data = {
            'image_width': int(self.image_shape[0]),
            'image_height': int(self.image_shape[1]),
            'camera_name': 'thermal_camera',
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': K.flatten().tolist()
            },
            'distortion_model': 'plumb_bob',
            'distortion_coefficients': {
                'rows': 1,
                'cols': len(dist.flatten()),
                'data': dist.flatten().tolist()
            },
            'rectification_matrix': {
                'rows': 3,
                'cols': 3,
                'data': np.eye(3).flatten().tolist()
            },
            'projection_matrix': {
                'rows': 3,
                'cols': 4,
                'data': np.hstack((K, np.zeros((3, 1)))).flatten().tolist()
            }
        }

        yaml_path = os.path.join(os.getcwd(), self.output_yaml)
        with open(yaml_path, 'w') as f:
            yaml.dump(calib_data, f)

        msg = f"Calibration complete. Saved to {yaml_path}"
        self.get_logger().info(msg)
        response.success = True
        response.message = msg
        return response

    # ----------------------------------------------------------
    # Service: clear stored samples
    # ----------------------------------------------------------
    def handle_clear_samples(self, request, response):
        self.objpoints.clear()
        self.imgpoints.clear()
        msg = "Cleared stored calibration samples."
        self.get_logger().info(msg)
        response.success = True
        response.message = msg
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ThermalCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()