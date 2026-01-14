import os
from datetime import datetime

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


class LivoxScanSaver(Node):
    def __init__(self):
        super().__init__('livox_scan_saver')

        # Parameters
        self.declare_parameter('topic', '/livox/scan_window')
        self.declare_parameter('output_root', '/home/semesterproject/Time/livox_scans')
        self.declare_parameter('save_once', True)  # if True: save first scan then stop

        self.topic = self.get_parameter('topic').get_parameter_value().string_value
        self.output_root = self.get_parameter('output_root').get_parameter_value().string_value
        self.save_once = self.get_parameter('save_once').get_parameter_value().bool_value

        os.makedirs(self.output_root, exist_ok=True)
        self.saved_count = 0

        self.sub = self.create_subscription(
            PointCloud2,
            self.topic,
            self.scan_callback,
            10
        )

        self.get_logger().info(
            f"LivoxScanSaver started.\n"
            f"  Listening on: {self.topic}\n"
            f"  Saving to: {self.output_root}\n"
            f"  save_once: {self.save_once}"
        )

    def scan_callback(self, msg: PointCloud2):
        # Build filename with timestamp and index
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"livox_scan_{self.saved_count:04d}_{now}.pcd"
        path = os.path.join(self.output_root, filename)

        # Read points (expects fields x, y, z, intensity from livox_accumulator)
        points = list(pc2.read_points(
            msg,
            field_names=("x", "y", "z", "intensity"),
            skip_nans=True
        ))
        num_points = len(points)

        # Write ASCII PCD file
        try:
            with open(path, 'w') as f:
                f.write("# .PCD v0.7\n")
                f.write("VERSION 0.7\n")
                f.write("FIELDS x y z intensity\n")
                f.write("SIZE 4 4 4 4\n")
                f.write("TYPE F F F F\n")
                f.write("COUNT 1 1 1 1\n")
                f.write(f"WIDTH {num_points}\n")
                f.write("HEIGHT 1\n")
                f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {num_points}\n")
                f.write("DATA ascii\n")
                for x, y, z, intensity in points:
                    f.write(f"{x} {y} {z} {intensity}\n")

            self.get_logger().info(
                f"Saved Livox scan #{self.saved_count} "
                f"({num_points} points) to: {path}"
            )
            self.saved_count += 1

            if self.save_once:
                self.get_logger().info("save_once=True → shutting down node.")
                # Let rclpy.spin() return
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Failed to write PCD file: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LivoxScanSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()