#!/usr/bin/env python3
"""Diagnostic: grab one point cloud and run it through funmap's actual
height-image pipeline to find where data gets lost."""

import sys
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import ros2_numpy as rn
import tf2_ros
from rclpy.time import Time
from rclpy.duration import Duration

class PCDebug(Node):
    def __init__(self):
        super().__init__("pc_debug")
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)
        self.pc_msg = None
        self.create_subscription(PointCloud2, "/camera/depth/color/points", self._cb, 1)

    def _cb(self, msg):
        if self.pc_msg is None:
            self.pc_msg = msg

def main():
    rclpy.init()
    node = PCDebug()

    print("Waiting for point cloud...")
    while node.pc_msg is None:
        rclpy.spin_once(node, timeout_sec=0.5)
    msg = node.pc_msg
    print(f"Got point cloud: frame={msg.header.frame_id}, "
          f"width={msg.width}, height={msg.height}, "
          f"point_step={msg.point_step}")

    pc = rn.numpify(msg)
    xyz = rn.point_cloud2.get_xyz_points(pc)
    valid = ~np.isnan(xyz).any(axis=1)
    print(f"Points: {len(xyz)} total, {valid.sum()} valid")

    vxyz = xyz[valid]
    print(f"Camera frame ranges: X=[{vxyz[:,0].min():.2f},{vxyz[:,0].max():.2f}] "
          f"Y=[{vxyz[:,1].min():.2f},{vxyz[:,1].max():.2f}] "
          f"Z=[{vxyz[:,2].min():.2f},{vxyz[:,2].max():.2f}]")

    # TF lookup
    frame_id = msg.header.frame_id
    mat = None
    for target in ["map", "odom"]:
        try:
            tf = node.tf_buf.lookup_transform(target, frame_id, Time(), Duration(seconds=2.0))
            mat = rn.numpify(tf.transform)
            print(f"TF {target}->{frame_id} OK")
            break
        except Exception as e:
            print(f"TF {target} failed: {e}")
    if mat is None:
        print("No TF available, cannot continue")
        return

    # Now try funmap's actual code path
    print("\n=== Testing funmap's actual pipeline ===")
    try:
        from stretch_funmap import ros_max_height_image as rm
        from stretch_funmap import mapping as ma
        from stretch_funmap import numba_height_image as nh
    except ImportError as e:
        print(f"Cannot import stretch_funmap: {e}")
        print("Run on the robot where stretch_funmap is installed")
        return

    # Create HeadScan exactly like funmap does (voi_side_m=16.0)
    print("Creating HeadScan(voi_side_m=16.0)...")
    head_scan = ma.HeadScan(voi_side_m=16.0)
    mhi = head_scan.max_height_im
    voi = mhi.voi
    print(f"VOI: frame={voi.frame_id}, origin={voi.origin}, "
          f"size=({voi.x_in_m},{voi.y_in_m},{voi.z_in_m})")
    print(f"Image shape: {mhi.image.shape}, dtype: {mhi.image.dtype}")
    print(f"m_per_pix: {mhi.m_per_pix}")
    print(f"Image non-zero before: {np.count_nonzero(mhi.image)}")

    # Step 1: get_points_to_voi_matrix_with_tf2
    print("\nStep 1: get_points_to_voi_matrix_with_tf2...")
    points_to_voi_mat, timestamp = voi.get_points_to_voi_matrix_with_tf2(
        frame_id, node.tf_buf)
    if points_to_voi_mat is None:
        print("FAILED — TF lookup returned None")
        print("This is why the height map is all zeros!")
        return
    print(f"OK — got 4x4 transform matrix")
    print(f"Matrix:\n{points_to_voi_mat}")

    # Step 2: split_rgb_field
    print("\nStep 2: split_rgb_field...")
    rgb_points = rn.point_cloud2.split_rgb_field(pc)
    print(f"rgb_points shape: {rgb_points.shape}")

    # Step 3: from_rgb_points (what actually writes to the height image)
    print("\nStep 3: from_rgb_points...")
    mhi.from_rgb_points(points_to_voi_mat, rgb_points)
    nonzero_after = np.count_nonzero(mhi.image)
    print(f"Image non-zero after: {nonzero_after}")

    if nonzero_after == 0:
        print("\n*** STILL ALL ZEROS — bug is in from_rgb_points / numba code ***")
        # Debug: manually transform a few points and check bounds
        sample = vxyz[:5]
        ones = np.ones((5, 1))
        pts_h = np.hstack([sample, ones])
        pts_voi = (points_to_voi_mat @ pts_h.T).T
        print(f"Sample points in VOI coords:")
        for i, p in enumerate(pts_voi):
            print(f"  [{i}] x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f}")
        print(f"VOI bounds: x=[0,{voi.x_in_m}] y=[-{voi.y_in_m},0] z=[0,{voi.z_in_m}]")
    else:
        print(f"\nSUCCESS — {nonzero_after} non-zero pixels in height image")
        print("Funmap pipeline works correctly with this point cloud.")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
