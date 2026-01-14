import struct

import rclpy
from rclpy.node import Node

from livox_ros_driver2.msg import CustomMsg
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


class LivoxScanAccumulator(Node):
    """
    Accumulate Livox /livox/lidar CustomMsg into a dense PointCloud2
    over a sliding time window (window_duration seconds).
    """

    def __init__(self):
        super().__init__("livox_scan_accumulator")

        # -------- Parameters --------
        self.declare_parameter("window_duration", 5.0)
        self.window_duration = (
            self.get_parameter("window_duration")
            .get_parameter_value()
            .double_value
        )

        self.get_logger().info(
            f"LivoxScanAccumulator: window_duration = {self.window_duration:.2f} s"
        )

        # -------- State --------
        self.window_start_time = None
        self.points = []  # list of (x, y, z, intensity)
        self.current_frame_id = "livox_frame"

        # -------- ROS interfaces --------
        self.sub = self.create_subscription(
            CustomMsg,
            "/livox/lidar",
            self.lidar_callback,
            10,
        )
        self.pub = self.create_publisher(
            PointCloud2,
            "/livox/scan_window",
            10,
        )

        self.get_logger().info("LivoxScanAccumulator node initialized and subscribed to /livox/lidar.")

    # ------------------------------------------------------
    # Callback
    # ------------------------------------------------------
    def lidar_callback(self, msg: CustomMsg):
        """
        Called for every Livox CustomMsg.
        We accumulate all points into a buffer until window_duration is reached,
        then publish a single dense PointCloud2 and reset.
        """
        now_ros = self.get_clock().now()

        if self.window_start_time is None:
            self.window_start_time = now_ros
            self.points = []
            self.get_logger().info(
                f"Started accumulation at t0 = {self.window_start_time.nanoseconds / 1e9:.3f}s"
            )

        # Save the frame_id so our published cloud matches the Livox frame.
        if msg.header.frame_id:
            self.current_frame_id = msg.header.frame_id

        # Append all points from this packet
        count_before = len(self.points)
        for p in msg.points:
            # reflectivity is uint8; cast to float for PointCloud2 "intensity"
            self.points.append((p.x, p.y, p.z, float(p.reflectivity)))
        added = len(self.points) - count_before

        # Some light logging so we know callbacks fire
        self.get_logger().debug(
            f"Received CustomMsg: point_num={msg.point_num}, added={added}, total={len(self.points)}"
        )

        # Check elapsed time in seconds
        elapsed = (now_ros - self.window_start_time).nanoseconds / 1e9

        if elapsed >= self.window_duration:
            self.get_logger().info(
                f"Accumulation done ({elapsed:.2f} s). Total points: {len(self.points)}"
            )
            self.publish_scan()
            # Reset for next window
            self.window_start_time = None
            self.points = []

    # ------------------------------------------------------
    # Publishing
    # ------------------------------------------------------
    def publish_scan(self):
        if not self.points:
            self.get_logger().warn("No points accumulated, skipping publish.")
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.current_frame_id

        # Define PointCloud2 fields: x, y, z, intensity
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        point_step = 16  # 4 floats * 4 bytes

        # Pack points into binary data
        data = bytearray()
        for x, y, z, intensity in self.points:
            data.extend(struct.pack("ffff", x, y, z, intensity))

        width = len(self.points)
        cloud = PointCloud2(
            header=header,
            height=1,
            width=width,
            fields=fields,
            is_bigendian=False,
            point_step=point_step,
            row_step=point_step * width,
            data=bytes(data),
            is_dense=True,
        )

        self.pub.publish(cloud)
        self.get_logger().info(
            f"Published dense scan on /livox/scan_window (height={cloud.height}, width={cloud.width})."
        )


def main(args=None):
    rclpy.init(args=args)
    node = LivoxScanAccumulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
