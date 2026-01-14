#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge

from message_filters import Subscriber, ApproximateTimeSynchronizer

import tf2_ros
from tf2_ros import TransformException
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from sensor_msgs_py import point_cloud2


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-12 else (v / n)


def ransac_plane(points_xyz: np.ndarray, iters: int, thresh: float, min_inliers: int):
    """
    Fit plane n·p + d = 0 with RANSAC. Returns (n, d, mask) or None.
    """
    N = points_xyz.shape[0]
    if N < 200:
        return None

    rng = np.random.default_rng()
    best_inliers = 0
    best_n = None
    best_d = None
    best_mask = None

    for _ in range(iters):
        idx = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = points_xyz[idx[0]], points_xyz[idx[1]], points_xyz[idx[2]]

        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            continue
        n = n / n_norm
        d = -float(np.dot(n, p1))

        dist = np.abs(points_xyz @ n + d)
        mask = dist < thresh
        inliers = int(np.sum(mask))

        if inliers > best_inliers:
            best_inliers = inliers
            best_n, best_d, best_mask = n, d, mask

    if best_n is None or best_inliers < min_inliers:
        return None

    # refine using inliers (SVD)
    inlier_pts = points_xyz[best_mask]
    centroid = np.mean(inlier_pts, axis=0)
    X = inlier_pts - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1, :]
    n = normalize(n)
    d = -float(np.dot(n, centroid))

    dist = np.abs(points_xyz @ n + d)
    mask = dist < thresh
    return n, d, mask


def make_plane_basis(n: np.ndarray):
    n = normalize(n)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = normalize(np.cross(n, a))
    e2 = normalize(np.cross(n, e1))
    return e1, e2


def point_in_convex_quad(pt2, quad2):
    x, y = pt2
    sign = None
    for i in range(4):
        x1, y1 = quad2[i]
        x2, y2 = quad2[(i + 1) % 4]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        s = cross >= 0
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


class LidarCameraROIFusion(Node):
    def __init__(self):
        super().__init__("lidar_camera_roi_fusion")

        # Topics / frames
        self.declare_parameter("cloud_topic", "/livox/scan_window")
        self.declare_parameter("image_topic", "/flir_camera/image_raw")
        self.declare_parameter("camera_info_topic", "/flir_camera/camera_info")
        self.declare_parameter("camera_frame", "flir_camera")  # IMPORTANT: your CameraInfo frame_id

        # ROI pixels [u_min, v_min, u_max, v_max]
        self.declare_parameter("roi", [400, 250, 1050, 900])

        # RANSAC / gating (meters)
        self.declare_parameter("ransac_iters", 250)
        self.declare_parameter("ransac_thresh_m", 0.03)
        self.declare_parameter("plane_thickness_m", 0.05)
        self.declare_parameter("min_inliers", 500)

        # debug draw
        self.declare_parameter("max_draw_points", 6000)

        self.cloud_topic = self.get_parameter("cloud_topic").value
        self.image_topic = self.get_parameter("image_topic").value
        self.caminfo_topic = self.get_parameter("camera_info_topic").value
        self.camera_frame = self.get_parameter("camera_frame").value

        self.bridge = CvBridge()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera intrinsics
        self.K = None
        self.width = None
        self.height = None
        self.create_subscription(CameraInfo, self.caminfo_topic, self.on_caminfo, 10)

        # Sync cloud + image
        self.sub_cloud = Subscriber(self, PointCloud2, self.cloud_topic)
        self.sub_img = Subscriber(self, Image, self.image_topic)

        self.sync = ApproximateTimeSynchronizer([self.sub_cloud, self.sub_img],
                                               queue_size=10, slop=0.10)
        self.sync.registerCallback(self.on_sync)

        self.pub_roi_cloud = self.create_publisher(PointCloud2, "/livox/cloud_roi_cam", 10)
        self.pub_debug_img = self.create_publisher(Image, "/debug/lidar_projection", 10)

        self.get_logger().info("ROI fusion node started.")

    def on_caminfo(self, msg: CameraInfo):
        if self.K is None:
            self.width = msg.width
            self.height = msg.height
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            fx = self.K[0, 0]
            fy = self.K[1, 1]
            cx = self.K[0, 2]
            cy = self.K[1, 2]
            self.get_logger().info(
                f"CameraInfo OK: w={self.width} h={self.height} fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}"
            )

    def on_sync(self, cloud_msg: PointCloud2, img_msg: Image):
        if self.K is None:
            return

        # TF: cloud frame -> camera frame
        try:
            tf = self.tf_buffer.lookup_transform(
                self.camera_frame,
                cloud_msg.header.frame_id,
                cloud_msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return

        cloud_cam = do_transform_cloud(cloud_msg, tf)

        # PointCloud2 -> numpy
        pts = []
        for p in point_cloud2.read_points(cloud_cam, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        if not pts:
            self.get_logger().warn("No points in cloud (after TF).")
            return
        pts = np.array(pts, dtype=np.float64)

        # Keep points in front of camera
        pts = pts[pts[:, 2] > 0.01]
        if pts.shape[0] < 200:
            self.get_logger().warn("Too few front points.")
            return

        # Fit plane
        iters = int(self.get_parameter("ransac_iters").value)
        thresh = float(self.get_parameter("ransac_thresh_m").value)
        min_inliers = int(self.get_parameter("min_inliers").value)

        model = ransac_plane(pts, iters=iters, thresh=thresh, min_inliers=min_inliers)
        if model is None:
            self.get_logger().warn("Plane fit failed: not enough inliers.")
            return
        n, d, _ = model

        # Build ROI footprint intersection with plane
        umin, vmin, umax, vmax = map(int, self.get_parameter("roi").value)

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        corners = [(umin, vmin), (umax, vmin), (umax, vmax), (umin, vmax)]
        quad3 = []
        for (u, v) in corners:
            x = (u - cx) / fx
            y = (v - cy) / fy
            ray = normalize(np.array([x, y, 1.0], dtype=np.float64))
            denom = float(np.dot(n, ray))
            if abs(denom) < 1e-9:
                self.get_logger().warn("ROI ray ~parallel to plane.")
                return
            t = -d / denom
            if t <= 0:
                self.get_logger().warn("ROI-plane intersection behind camera.")
                return
            quad3.append(t * ray)
        quad3 = np.array(quad3, dtype=np.float64)

        # Gate points near plane
        plane_thick = float(self.get_parameter("plane_thickness_m").value)
        dist = np.abs(pts @ n + d)
        pts_near = pts[dist < plane_thick]
        if pts_near.shape[0] < 50:
            self.get_logger().warn("Too few near-plane points.")
            return

        # Convert to plane 2D and inside-quad test
        e1, e2 = make_plane_basis(n)

        def proj2(p3):
            return np.array([np.dot(e1, p3), np.dot(e2, p3)], dtype=np.float64)

        quad2 = np.vstack([proj2(q) for q in quad3])
        pts2 = np.vstack([proj2(p) for p in pts_near])

        inside_mask = np.array([point_in_convex_quad(pts2[i], quad2) for i in range(pts2.shape[0])], dtype=bool)
        pts_roi = pts_near[inside_mask]

        # Publish ROI cloud in camera frame
        header = cloud_cam.header
        roi_cloud_msg = point_cloud2.create_cloud_xyz32(header, pts_roi.astype(np.float32).tolist())
        self.pub_roi_cloud.publish(roi_cloud_msg)

        # Debug projection
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        inside_count = 0
        if pts_roi.shape[0] > 0:
            max_draw = int(self.get_parameter("max_draw_points").value)
            if pts_roi.shape[0] > max_draw:
                idx = np.random.choice(pts_roi.shape[0], size=max_draw, replace=False)
                draw_pts = pts_roi[idx]
            else:
                draw_pts = pts_roi

            X, Y, Z = draw_pts[:, 0], draw_pts[:, 1], draw_pts[:, 2]
            u = (fx * X / Z + cx).astype(np.int32)
            v = (fy * Y / Z + cy).astype(np.int32)

            inside = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
            inside_count = int(np.sum(inside))

            for ui, vi in zip(u[inside], v[inside]):
                cv_img[vi, ui] = (0, 255, 0)

        dbg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
        dbg.header = img_msg.header
        self.pub_debug_img.publish(dbg)

        self.get_logger().info(
            f"ROI pts: {pts_roi.shape[0]} | projected inside: {inside_count} | ROI px [{umin},{vmin}]→[{umax},{vmax}]"
        )


def main():
    rclpy.init()
    node = LidarCameraROIFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()