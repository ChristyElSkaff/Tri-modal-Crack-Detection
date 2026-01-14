#!/usr/bin/env python3
import json
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge


class CollectPairs(Node):
    def __init__(self):
        super().__init__("collect_pairs")

        self.declare_parameter("image_topic", "/flir_camera/image_raw")
        self.declare_parameter("clicked_topic", "/clicked_point")
        self.declare_parameter("out_path", "/home/semesterproject/Time/pairs_lidar_cam.json")

        self.image_topic = self.get_parameter("image_topic").value
        self.clicked_topic = self.get_parameter("clicked_topic").value
        self.out_path = self.get_parameter("out_path").value

        self.bridge = CvBridge()
        self.last_img = None
        self.last_point = None
        self.last_point_time = 0.0

        self.pairs = []  # list of dicts: {X,Y,Z,u,v,frame_id}

        self.create_subscription(Image, self.image_topic, self.on_image, 10)
        self.create_subscription(PointStamped, self.clicked_topic, self.on_clicked, 10)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("image", self.on_mouse)

        self.get_logger().info(
            "Collector running.\n"
            "Workflow:\n"
            "  1) In RViz, click a LiDAR point (Publish Point)\n"
            "  2) Immediately click the corresponding pixel here\n"
            "Keys:\n"
            "  s = save JSON\n"
            "  u = undo last pair\n"
            "  q = quit\n"
        )

    def on_image(self, msg: Image):
        self.last_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # draw existing points
        img = self.last_img.copy()
        for p in self.pairs:
            cv2.circle(img, (int(p["u"]), int(p["v"])), 4, (0, 255, 0), -1)

        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            self.save()
        elif key == ord("u"):
            if self.pairs:
                self.pairs.pop()
                self.get_logger().info(f"Undo. Remaining pairs: {len(self.pairs)}")
        elif key == ord("q"):
            self.save()
            raise KeyboardInterrupt

    def on_clicked(self, msg: PointStamped):
        self.last_point = msg
        self.last_point_time = time.time()
        self.get_logger().info(
            f"Got clicked_point in frame '{msg.header.frame_id}': "
            f"({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})"
        )

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.last_img is None:
            return
        if self.last_point is None or (time.time() - self.last_point_time) > 10.0:
            self.get_logger().warn("No recent /clicked_point. Click a LiDAR point in RViz first.")
            return

        p = self.last_point
        pair = {
            "frame_id": p.header.frame_id,
            "X": float(p.point.x),
            "Y": float(p.point.y),
            "Z": float(p.point.z),
            "u": int(x),
            "v": int(y),
        }
        self.pairs.append(pair)
        self.get_logger().info(f"Added pair #{len(self.pairs)}: 3D=({pair['X']:.3f},{pair['Y']:.3f},{pair['Z']:.3f}) px=({pair['u']},{pair['v']})")

    def save(self):
        with open(self.out_path, "w") as f:
            json.dump(self.pairs, f, indent=2)
        self.get_logger().info(f"Saved {len(self.pairs)} pairs to {self.out_path}")


def main():
    rclpy.init()
    node = CollectPairs()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()