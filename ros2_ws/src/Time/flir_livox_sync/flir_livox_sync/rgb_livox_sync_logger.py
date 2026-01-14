#!/usr/bin/env python3
import os
from datetime import datetime

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


def _has_field(cloud_msg: PointCloud2, name: str) -> bool:
    return any(f.name == name for f in cloud_msg.fields)


def save_pcd_ascii(cloud_msg: PointCloud2, out_path: str):
    """
    Save PointCloud2 to a simple ASCII PCD file.
    Writes x y z and (if present) intensity.
    """
    has_intensity = _has_field(cloud_msg, "intensity")
    field_names = ["x", "y", "z"] + (["intensity"] if has_intensity else [])

    points = list(pc2.read_points(cloud_msg, field_names=field_names, skip_nans=True))
    n = len(points)

    with open(out_path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write(f"FIELDS {' '.join(field_names)}\n")
        f.write(f"SIZE {' '.join(['4'] * len(field_names))}\n")
        f.write(f"TYPE {' '.join(['F'] * len(field_names))}\n")
        f.write(f"COUNT {' '.join(['1'] * len(field_names))}\n")
        f.write(f"WIDTH {n}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write("DATA ascii\n")
        for p in points:
            f.write(" ".join(str(float(v)) for v in p) + "\n")


class RgbLivoxSyncSaver(Node):
    def __init__(self):
        super().__init__("rgb_livox_sync_saver")
        self.bridge = CvBridge()
        self.count = 0

        # -------- Parameters --------
        self.declare_parameter("rgb_topic", "/flir_camera/image_raw")
        self.declare_parameter("cloud_topic", "/livox/scan_window")  # your scan window
        self.declare_parameter("slop", 0.10)                         # seconds
        self.declare_parameter("queue_size", 30)
        self.declare_parameter("output_root", "/home/semesterproject/Time/synced_data")

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.cloud_topic = self.get_parameter("cloud_topic").value
        self.slop = float(self.get_parameter("slop").value)
        self.queue_size = int(self.get_parameter("queue_size").value)
        self.output_root = self.get_parameter("output_root").value

        # -------- Output folders --------
        session = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.output_root, session)
        self.rgb_dir = os.path.join(self.session_dir, "rgb")
        self.cloud_dir = os.path.join(self.session_dir, "lidar")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.cloud_dir, exist_ok=True)

        # -------- Sync subscribers --------
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
        self.cloud_sub = Subscriber(self, PointCloud2, self.cloud_topic)

        self.sync = ApproximateTimeSynchronizer(
            fs=[self.rgb_sub, self.cloud_sub],
            queue_size=self.queue_size,
            slop=self.slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info("RGB+Livox sync saver started")
        self.get_logger().info(f"  RGB topic   : {self.rgb_topic}")
        self.get_logger().info(f"  Cloud topic : {self.cloud_topic} (scan window)")
        self.get_logger().info(f"  slop        : {self.slop}s, queue_size: {self.queue_size}")
        self.get_logger().info(f"  output dir  : {self.session_dir}")

    def synced_callback(self, img_msg: Image, cloud_msg: PointCloud2):
        idx = self.count
        self.count += 1

        # --- Save RGB ---
        try:
            import cv2
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            rgb_path = os.path.join(self.rgb_dir, f"rgb_{idx:06d}.png")
            cv2.imwrite(rgb_path, cv_img)
        except Exception as e:
            self.get_logger().error(f"Failed to save RGB: {e}")
            return

        # --- Save PCD ---
        try:
            cloud_path = os.path.join(self.cloud_dir, f"cloud_{idx:06d}.pcd")
            save_pcd_ascii(cloud_msg, cloud_path)
        except Exception as e:
            self.get_logger().error(f"Failed to save PCD: {e}")
            return

        self.get_logger().info(
            f"Saved pair #{idx:06d} -> {os.path.basename(rgb_path)}, {os.path.basename(cloud_path)}"
        )


def main():
    rclpy.init()
    node = RgbLivoxSyncSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()