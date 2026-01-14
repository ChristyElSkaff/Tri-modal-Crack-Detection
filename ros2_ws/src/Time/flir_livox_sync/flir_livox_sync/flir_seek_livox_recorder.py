import os
import csv
from datetime import datetime

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


class FlirSeekLivoxSyncLogger(Node):
    def __init__(self):
        super().__init__('flir_seek_livox_sync_logger')

        # --- Parameters ---
        self.declare_parameter('output_root', '/home/Time/ros2_data')
        self.declare_parameter('slop', 0.5)          # seconds
        self.declare_parameter('save_seek_npy', True)  # save raw temps as .npy
        self.declare_parameter('seek_colormap', 'inferno')  # inferno|jet|turbo|hot|none

        output_root = self.get_parameter('output_root').get_parameter_value().string_value
        slop = self.get_parameter('slop').get_parameter_value().double_value
        self.save_seek_npy = self.get_parameter('save_seek_npy').get_parameter_value().bool_value
        self.seek_colormap = self.get_parameter('seek_colormap').get_parameter_value().string_value.lower()

        # Create a session directory with timestamp
        session_name = datetime.now().strftime('session_%Y%m%d_%H%M%S_synced')
        self.session_dir = os.path.join(output_root, session_name)
        self.flir_dir = os.path.join(self.session_dir, 'flir')
        self.seek_dir = os.path.join(self.session_dir, 'seek')
        self.lidar_dir = os.path.join(self.session_dir, 'lidar_pcd')

        for d in [self.session_dir, self.flir_dir, self.seek_dir, self.lidar_dir]:
            os.makedirs(d, exist_ok=True)

        # Metadata log file
        self.meta_path = os.path.join(self.session_dir, 'sync_log.csv')
        with open(self.meta_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'index',
                'flir_stamp_sec', 'flir_stamp_nsec',
                'seek_stamp_sec', 'seek_stamp_nsec',
                'lidar_stamp_sec', 'lidar_stamp_nsec',
                'dt_flir_ms', 'dt_seek_ms',
                'flir_file',
                'seek_vis_file',
                'seek_temp_file',   # <-- raw temps (npy)
                'lidar_pcd_file',
                'num_points'
            ])

        self.bridge = CvBridge()
        self.frame_index = 0

        # --- Subscribers & synchronizer ---
        self.flir_sub = Subscriber(self, Image, '/flir_camera/image_raw')
        self.seek_sub = Subscriber(self, Image, '/seek_thermal/image_raw')
        self.lidar_sub = Subscriber(self, PointCloud2, '/livox/scan_window')

        queue_size = 50
        self.sync = ApproximateTimeSynchronizer(
            [self.flir_sub, self.seek_sub, self.lidar_sub],
            queue_size,
            slop
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info(
            'FlirSeekLivoxSyncLogger started.\n'
            f'  Output root: {output_root}\n'
            f'  Session dir: {self.session_dir}\n'
            '  Subscribing to:\n'
            '    /flir_camera/image_raw\n'
            '    /seek_thermal/image_raw\n'
            '    /livox/scan_window (PointCloud2)\n'
            f'  ApproximateTimeSynchronizer slop={slop:.3f} s\n'
            f'  Save Seek NPY: {self.save_seek_npy}\n'
            f'  Seek colormap: {self.seek_colormap}'
        )

    def _colormap_id(self, cv2):
        # Map string -> OpenCV colormap
        if self.seek_colormap == 'inferno':
            return cv2.COLORMAP_INFERNO
        if self.seek_colormap == 'jet':
            return cv2.COLORMAP_JET
        if self.seek_colormap == 'turbo':
            return cv2.COLORMAP_TURBO
        if self.seek_colormap == 'hot':
            return cv2.COLORMAP_HOT
        return None  # 'none' or unknown

    def synced_callback(self, flir_msg: Image, seek_msg: Image, scan_msg: PointCloud2):
        # Timestamps
        tf = flir_msg.header.stamp
        ts = seek_msg.header.stamp
        tl = scan_msg.header.stamp

        t_flir = tf.sec + tf.nanosec * 1e-9
        t_seek = ts.sec + ts.nanosec * 1e-9
        t_lidar = tl.sec + tl.nanosec * 1e-9

        dt_flir_ms = (t_flir - t_lidar) * 1000.0
        dt_seek_ms = (t_seek - t_lidar) * 1000.0

        idx = self.frame_index
        base = f'{idx:06d}'

        # ----------------------------
        # Save FLIR image (PNG, BGR)
        # ----------------------------
        flir_file = ''
        try:
            import cv2
            flir_cv = self.bridge.imgmsg_to_cv2(flir_msg, desired_encoding='bgr8')
            flir_file = f'flir_{base}.png'
            flir_path = os.path.join(self.flir_dir, flir_file)
            cv2.imwrite(flir_path, flir_cv)
        except Exception as e:
            self.get_logger().warn(f'Failed to save FLIR image: {e}')
            flir_file = ''

        # ---------------------------------------------
        # Save Seek thermal:
        #   1) seek_vis_file: PNG visualization
        #   2) seek_temp_file: NPY float32 temps (°C)
        # ---------------------------------------------
        seek_vis_file = ''
        seek_temp_file = ''
        try:
            import cv2
            import numpy as np

            # Raw thermal data (should be float32 2D for encoding "32FC1")
            seek_cv = self.bridge.imgmsg_to_cv2(seek_msg, desired_encoding='passthrough')

            # Ensure float32 temps matrix
            seek_temp = seek_cv.astype(np.float32, copy=False)

            # 1) Save raw temperature matrix (.npy)
            if self.save_seek_npy:
                seek_temp_file = f'seek_{base}.npy'
                seek_temp_path = os.path.join(self.seek_dir, seek_temp_file)
                np.save(seek_temp_path, seek_temp)

            # 2) Create a visualization PNG (normalize to 0..255)
            #    Note: this is ONLY for viewing; temps are in the .npy
            temp_min = float(np.nanmin(seek_temp))
            temp_max = float(np.nanmax(seek_temp))

            if temp_max > temp_min:
                norm = (seek_temp - temp_min) / (temp_max - temp_min)
            else:
                norm = np.zeros_like(seek_temp, dtype=np.float32)

            seek_u8 = (norm * 255.0).astype('uint8')

            cmap = self._colormap_id(cv2)
            if cmap is not None:
                seek_vis = cv2.applyColorMap(seek_u8, cmap)  # BGR color
            else:
                seek_vis = seek_u8  # grayscale

            seek_vis_file = f'seek_{base}.png'
            seek_vis_path = os.path.join(self.seek_dir, seek_vis_file)
            cv2.imwrite(seek_vis_path, seek_vis)

        except Exception as e:
            self.get_logger().warn(f'Failed to save Seek image/temps: {e}')
            seek_vis_file = ''
            seek_temp_file = ''

        # ----------------------------
        # Save LiDAR scan as PCD (ASCII)
        # ----------------------------
        lidar_pcd_file = f'lidar_{base}.pcd'
        lidar_pcd_path = os.path.join(self.lidar_dir, lidar_pcd_file)
        num_points = 0

        try:
            points = []
            for x, y, z, intensity in pc2.read_points(
                scan_msg,
                field_names=("x", "y", "z", "intensity"),
                skip_nans=True
            ):
                points.append((float(x), float(y), float(z), float(intensity)))

            num_points = len(points)

            with open(lidar_pcd_path, 'w') as f:
                f.write("# .PCD v0.7 - Point Cloud Data file format\n")
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

        except Exception as e:
            self.get_logger().warn(f'Failed to save LiDAR PCD: {e}')
            lidar_pcd_file = ''
            num_points = 0

        # ----------------------------
        # Append to metadata log
        # ----------------------------
        try:
            with open(self.meta_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx,
                    tf.sec, tf.nanosec,
                    ts.sec, ts.nanosec,
                    tl.sec, tl.nanosec,
                    f'{dt_flir_ms:.3f}', f'{dt_seek_ms:.3f}',
                    flir_file,
                    seek_vis_file,
                    seek_temp_file,
                    lidar_pcd_file,
                    num_points
                ])
        except Exception as e:
            self.get_logger().warn(f'Failed to write sync_log row: {e}')

        self.get_logger().info(
            f'Saved synced triple #{idx}: '
            f'dt_flir={dt_flir_ms:+6.2f} ms, dt_seek={dt_seek_ms:+6.2f} ms, '
            f'points={num_points}'
        )

        self.frame_index += 1


def main(args=None):
    rclpy.init(args=args)
    node = FlirSeekLivoxSyncLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
