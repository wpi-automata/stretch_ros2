#!/usr/bin/env python3
"""Node 1: Mapping and Exploration.

Explores a room using frontier-based exploration and builds both:
  - A 3D voxel map (via stretch_funmap's max-height-image approach)
  - A 2D occupancy grid (published to /map for Nav2 and RViz)

Frontier selection can use either the 2D occupancy grid (default) or
the 3D voxel map, controlled by the `frontier_method` parameter.

The node publishes:
  - /map (nav_msgs/OccupancyGrid): 2D occupancy grid (shown in RViz Map display)
  - /voxel_map_cloud (sensor_msgs/PointCloud2): 3D voxel map as colored pointcloud
  - /voxel_markers (visualization_msgs/MarkerArray): 3D voxel cubes (alternative viz)
  - /frontier_markers (visualization_msgs/MarkerArray): current frontier cells
  - /exploration_status (std_msgs/String): current exploration state

The node provides:
  - /mapping/start (std_srvs/Trigger): begin exploration
  - /mapping/stop (std_srvs/Trigger): stop exploration early
  - /mapping/is_complete (std_srvs/Trigger): query if mapping finished

When exploration completes (no more frontiers), it publishes a final
map and sets the `mapping_complete` flag.
"""

import math
import threading
import time
from enum import Enum

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist, TransformStamped, Point
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String, Header, ColorRGBA
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import struct


class ExplorationState(Enum):
    IDLE = "idle"
    ROTATING = "rotating"
    PLANNING = "planning"
    NAVIGATING = "navigating"
    COMPLETE = "complete"


class MappingNode(Node):
    """Frontier-based exploration and mapping for Stretch3."""

    def __init__(self):
        super().__init__("mapping_node")

        # Parameters
        self.declare_parameter("frontier_method", "occupancy_grid")
        self.declare_parameter("map_resolution", 0.05)
        self.declare_parameter("map_width", 200)
        self.declare_parameter("map_height", 200)
        self.declare_parameter("voxel_resolution", 0.05)
        self.declare_parameter("exploration_timeout_s", 300.0)
        self.declare_parameter("min_frontier_size", 5)
        self.declare_parameter("robot_radius", 0.18)
        self.declare_parameter("max_exploration_attempts", 50)
        self.declare_parameter("use_sim", False)

        self.frontier_method = self.get_parameter("frontier_method").value
        self.map_resolution = self.get_parameter("map_resolution").value
        self.map_width = self.get_parameter("map_width").value
        self.map_height = self.get_parameter("map_height").value
        self.voxel_resolution = self.get_parameter("voxel_resolution").value
        self.exploration_timeout = self.get_parameter("exploration_timeout_s").value
        self.min_frontier_size = self.get_parameter("min_frontier_size").value
        self.robot_radius = self.get_parameter("robot_radius").value
        self.max_attempts = self.get_parameter("max_exploration_attempts").value
        self.use_sim = self.get_parameter("use_sim").value

        self.get_logger().info(
            f"Mapping node initialized: frontier_method={self.frontier_method}, "
            f"resolution={self.map_resolution}m"
        )

        # State
        self.state = ExplorationState.IDLE
        self.mapping_complete = False
        self.exploration_thread = None
        self.stop_requested = False

        # Map data
        self.occupancy_grid = np.full(
            (self.map_height, self.map_width), -1, dtype=np.int8
        )
        self.voxel_map = {}
        self.robot_pose = None
        self.map_origin_x = -(self.map_width * self.map_resolution) / 2.0
        self.map_origin_y = -(self.map_height * self.map_resolution) / 2.0

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Callback group for services
        self.cb_group = ReentrantCallbackGroup()

        # Publishers
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", latched_qos)
        self.voxel_cloud_pub = self.create_publisher(
            PointCloud2, "/voxel_map_cloud", 10
        )
        self.voxel_pub = self.create_publisher(
            MarkerArray, "/voxel_markers", 10
        )
        self.frontier_pub = self.create_publisher(
            MarkerArray, "/frontier_markers", 10
        )
        self.status_pub = self.create_publisher(String, "/exploration_status", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/stretch/cmd_vel", 10)
        self.goal_pub = self.create_publisher(
            PoseStamped, "/goal_pose", 10
        )

        # Subscribers — use BEST_EFFORT QoS to match the simulation driver
        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            PointCloud2,
            "/camera/depth/color/points",
            self.pointcloud_callback,
            sensor_qos,
        )

        # Services
        self.create_service(
            Trigger, "/mapping/start", self.start_callback,
            callback_group=self.cb_group
        )
        self.create_service(
            Trigger, "/mapping/stop", self.stop_callback,
            callback_group=self.cb_group
        )
        self.create_service(
            Trigger, "/mapping/is_complete", self.is_complete_callback,
            callback_group=self.cb_group
        )

        # Timer to publish status and map
        self.create_timer(1.0, self.publish_status)
        self.create_timer(2.0, self.publish_map)

    # ─── Callbacks ────────────────────────────────────────────────────

    def pointcloud_callback(self, msg: PointCloud2):
        """Integrate incoming pointcloud into occupancy grid and voxel map."""
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", msg.header.frame_id,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5)
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return

        self._update_robot_pose()
        self._integrate_pointcloud(msg, transform)

    def start_callback(self, request, response):
        """Start frontier exploration."""
        if self.state != ExplorationState.IDLE and self.state != ExplorationState.COMPLETE:
            response.success = False
            response.message = f"Already exploring (state={self.state.value})"
            return response

        self.mapping_complete = False
        self.stop_requested = False
        self.exploration_thread = threading.Thread(
            target=self._exploration_loop, daemon=True
        )
        self.exploration_thread.start()
        response.success = True
        response.message = "Exploration started"
        return response

    def stop_callback(self, request, response):
        """Stop exploration early."""
        self.stop_requested = True
        response.success = True
        response.message = "Stop requested"
        return response

    def is_complete_callback(self, request, response):
        """Query whether mapping is finished."""
        response.success = self.mapping_complete
        response.message = f"state={self.state.value}, complete={self.mapping_complete}"
        return response

    # ─── Exploration loop ─────────────────────────────────────────────

    def _exploration_loop(self):
        """Main exploration logic: rotate, find frontiers, navigate, repeat."""
        self.get_logger().info("Exploration loop started")
        start_time = time.time()
        attempt = 0

        # Initial 360-degree rotation to seed the map
        self._set_state(ExplorationState.ROTATING)
        self._rotate_360()

        while not self.stop_requested:
            elapsed = time.time() - start_time
            if elapsed > self.exploration_timeout:
                self.get_logger().info("Exploration timeout reached")
                break

            attempt += 1
            if attempt > self.max_attempts:
                self.get_logger().info("Max exploration attempts reached")
                break

            # Find frontiers
            self._set_state(ExplorationState.PLANNING)
            frontier_goal = self._find_best_frontier()

            if frontier_goal is None:
                self.get_logger().info("No more frontiers — exploration complete")
                break

            # Navigate to frontier
            self._set_state(ExplorationState.NAVIGATING)
            success = self._navigate_to_goal(frontier_goal)

            if not success:
                self.get_logger().warn(
                    f"Navigation to frontier failed (attempt {attempt})"
                )
                # Rotate to try to discover new frontiers
                self._set_state(ExplorationState.ROTATING)
                self._rotate_in_place(math.pi / 2)
                continue

            # Arrived — do a head sweep to gather more data
            self._set_state(ExplorationState.ROTATING)
            self._rotate_in_place(2 * math.pi)

        self._set_state(ExplorationState.COMPLETE)
        self.mapping_complete = True
        self.get_logger().info(
            f"Exploration complete: {attempt} attempts, "
            f"{time.time() - start_time:.1f}s elapsed"
        )

    # ─── Frontier detection ───────────────────────────────────────────

    def _find_best_frontier(self):
        """Find the best frontier cell to explore next.

        Returns (x, y) in world coordinates or None if no frontier exists.
        """
        if self.frontier_method == "voxel":
            return self._find_frontier_voxel()
        return self._find_frontier_occupancy()

    def _find_frontier_occupancy(self):
        """Find frontiers from the 2D occupancy grid.

        A frontier cell is free (0) and adjacent to at least one unknown (-1).
        """
        grid = self.occupancy_grid.copy()
        free_mask = (grid == 0)
        unknown_mask = (grid == -1)

        # Dilate unknown to find free cells touching unknown
        kernel = np.ones((3, 3), dtype=np.uint8)
        unknown_dilated = cv2.dilate(
            unknown_mask.astype(np.uint8), kernel, iterations=1
        )
        frontier_mask = free_mask & (unknown_dilated > 0)

        # Cluster frontier cells
        frontier_labels, n_labels = cv2.connectedComponents(
            frontier_mask.astype(np.uint8)
        )

        best_frontier = None
        best_score = -1.0

        robot_grid = self._world_to_grid(self.robot_pose) if self.robot_pose else None

        for label in range(1, n_labels + 1):
            cells = np.argwhere(frontier_labels == label)
            if len(cells) < self.min_frontier_size:
                continue

            centroid = cells.mean(axis=0)
            size = len(cells)

            if robot_grid is not None:
                dist = np.linalg.norm(centroid - robot_grid)
                # Score: prefer large nearby frontiers
                score = size / (dist + 1.0)
            else:
                score = float(size)

            if score > best_score:
                best_score = score
                best_frontier = centroid

        if best_frontier is None:
            return None

        return self._grid_to_world(best_frontier)

    def _find_frontier_voxel(self):
        """Find frontiers using the 3D voxel map.

        Projects occupied voxels to 2D, then finds boundaries between
        explored and unexplored regions.
        """
        if not self.voxel_map:
            return self._find_frontier_occupancy()

        # Project voxels to 2D occupancy
        projected = np.full((self.map_height, self.map_width), -1, dtype=np.int8)
        for (vx, vy, vz), occupied in self.voxel_map.items():
            gx = int((vx * self.voxel_resolution - self.map_origin_x) / self.map_resolution)
            gy = int((vy * self.voxel_resolution - self.map_origin_y) / self.map_resolution)
            if 0 <= gx < self.map_width and 0 <= gy < self.map_height:
                if occupied:
                    projected[gy, gx] = 100
                else:
                    if projected[gy, gx] != 100:
                        projected[gy, gx] = 0

        # Use same frontier logic on the projected grid
        old_grid = self.occupancy_grid
        self.occupancy_grid = projected
        result = self._find_frontier_occupancy()
        self.occupancy_grid = old_grid
        return result

    # ─── Map integration ──────────────────────────────────────────────

    def _integrate_pointcloud(self, msg: PointCloud2, transform: TransformStamped):
        """Integrate a point cloud into the occupancy grid and voxel map."""
        import sensor_msgs_py.point_cloud2 as pc2

        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if not points:
            return

        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        tz = transform.transform.translation.z

        from tf_transformations import quaternion_matrix
        q = transform.transform.rotation
        rot_mat = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

        for px, py, pz in points:
            # Transform point to map frame
            p_local = np.array([px, py, pz])
            p_map = rot_mat @ p_local + np.array([tx, ty, tz])

            # Update 2D occupancy grid (project to ground plane)
            gx = int((p_map[0] - self.map_origin_x) / self.map_resolution)
            gy = int((p_map[1] - self.map_origin_y) / self.map_resolution)
            if 0 <= gx < self.map_width and 0 <= gy < self.map_height:
                if p_map[2] > 0.05 and p_map[2] < 2.0:
                    self.occupancy_grid[gy, gx] = 100
                elif p_map[2] <= 0.05:
                    if self.occupancy_grid[gy, gx] != 100:
                        self.occupancy_grid[gy, gx] = 0

            # Update 3D voxel map
            vx = int(p_map[0] / self.voxel_resolution)
            vy = int(p_map[1] / self.voxel_resolution)
            vz = int(p_map[2] / self.voxel_resolution)
            is_occupied = p_map[2] > 0.05
            self.voxel_map[(vx, vy, vz)] = is_occupied

        # Raytrace free space from sensor origin to each point
        if self.robot_pose is not None:
            robot_grid = self._world_to_grid(self.robot_pose)
            if robot_grid is not None:
                rx, ry = int(robot_grid[0]), int(robot_grid[1])
                # Mark cells along rays as free (subsample for performance)
                for px, py, pz in points[::10]:
                    p_local = np.array([px, py, pz])
                    p_map = rot_mat @ p_local + np.array([tx, ty, tz])
                    gx = int((p_map[0] - self.map_origin_x) / self.map_resolution)
                    gy = int((p_map[1] - self.map_origin_y) / self.map_resolution)
                    self._raytrace_free(ry, rx, gy, gx)

    def _raytrace_free(self, y0, x0, y1, x1):
        """Bresenham ray from (y0,x0) to (y1,x1), marking cells as free."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        steps = 0
        max_steps = max(dx, dy)
        while steps < max_steps - 1:
            if 0 <= x0 < self.map_width and 0 <= y0 < self.map_height:
                if self.occupancy_grid[y0, x0] == -1:
                    self.occupancy_grid[y0, x0] = 0
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            steps += 1

    # ─── Navigation helpers ───────────────────────────────────────────

    def _navigate_to_goal(self, goal_xy):
        """Publish a goal pose and wait for the robot to reach it.

        Uses Nav2 goal_pose topic. Returns True if reached within tolerance.
        """
        if goal_xy is None:
            return False

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = float(goal_xy[0])
        goal_msg.pose.position.y = float(goal_xy[1])
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_msg)
        self.get_logger().info(
            f"Navigating to frontier ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})"
        )

        # Wait for robot to reach goal (poll pose)
        timeout = 30.0
        start = time.time()
        while time.time() - start < timeout and not self.stop_requested:
            self._update_robot_pose()
            if self.robot_pose is not None:
                dx = goal_xy[0] - self.robot_pose[0]
                dy = goal_xy[1] - self.robot_pose[1]
                if math.sqrt(dx * dx + dy * dy) < 0.3:
                    return True
            time.sleep(0.5)

        return False

    def _rotate_360(self):
        """Rotate 360 degrees in place to seed initial map."""
        self._rotate_in_place(2 * math.pi)

    def _rotate_in_place(self, angle_rad):
        """Rotate in place by publishing cmd_vel."""
        angular_speed = 0.5
        duration = abs(angle_rad) / angular_speed

        twist = Twist()
        twist.angular.z = angular_speed if angle_rad > 0 else -angular_speed

        start = time.time()
        while time.time() - start < duration and not self.stop_requested:
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)

        # Stop
        self.cmd_vel_pub.publish(Twist())
        time.sleep(0.5)

    def _update_robot_pose(self):
        """Get current robot pose from TF."""
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link",
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
            )
            self.robot_pose = (
                transform.transform.translation.x,
                transform.transform.translation.y,
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            pass

    # ─── Coordinate transforms ────────────────────────────────────────

    def _world_to_grid(self, world_xy):
        """Convert world (x, y) to grid (col, row)."""
        if world_xy is None:
            return None
        gx = int((world_xy[0] - self.map_origin_x) / self.map_resolution)
        gy = int((world_xy[1] - self.map_origin_y) / self.map_resolution)
        return np.array([gx, gy])

    def _grid_to_world(self, grid_rc):
        """Convert grid (row, col) to world (x, y)."""
        # grid_rc is (row, col) from numpy
        x = grid_rc[1] * self.map_resolution + self.map_origin_x
        y = grid_rc[0] * self.map_resolution + self.map_origin_y
        return (x, y)

    # ─── Publishing ───────────────────────────────────────────────────

    def publish_status(self):
        """Publish current exploration state."""
        msg = String()
        msg.data = self.state.value
        self.status_pub.publish(msg)

    def publish_map(self):
        """Publish the occupancy grid and all voxel visualizations."""
        now = self.get_clock().now().to_msg()

        # 2D Occupancy Grid (displayed via RViz "Map" display)
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = now
        msg.header.frame_id = "map"

        msg.info = MapMetaData()
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = self.occupancy_grid.flatten().tolist()
        self.map_pub.publish(msg)

        # 3D Voxel map as PointCloud2 (best RViz display for dense 3D data)
        self._publish_voxel_pointcloud(now)

        # 3D Voxel map as cube markers (alternative visualization)
        self._publish_voxel_markers(now)

        # Frontier cells visualization
        self._publish_frontier_markers(now)

    def _publish_voxel_pointcloud(self, stamp):
        """Publish the 3D voxel map as a colored PointCloud2.

        This is the best way to display dense voxel data in RViz:
        use the PointCloud2 display with color by axis (Z) or RGB.
        Each occupied voxel becomes a point colored by its height.
        """
        if not self.voxel_map:
            return

        points_data = []
        for (vx, vy, vz), occupied in self.voxel_map.items():
            if not occupied:
                continue
            x = float(vx) * self.voxel_resolution
            y = float(vy) * self.voxel_resolution
            z = float(vz) * self.voxel_resolution

            # Color by height: blue (low) → red (high)
            height_norm = min(max(z / 2.0, 0.0), 1.0)
            r = int(height_norm * 255)
            g = 80
            b = int((1.0 - height_norm) * 255)
            rgb_packed = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]

            points_data.append((x, y, z, rgb_packed))

        if not points_data:
            return

        # Build PointCloud2 message with XYZRGB fields
        cloud_msg = PointCloud2()
        cloud_msg.header.frame_id = "map"
        cloud_msg.header.stamp = stamp
        cloud_msg.height = 1
        cloud_msg.width = len(points_data)
        cloud_msg.is_dense = True
        cloud_msg.is_bigendian = False

        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.point_step = 16
        cloud_msg.row_step = cloud_msg.point_step * len(points_data)

        buf = bytearray()
        for x, y, z, rgb in points_data:
            buf += struct.pack('ffff', x, y, z, rgb)
        cloud_msg.data = bytes(buf)

        self.voxel_cloud_pub.publish(cloud_msg)

    def _publish_voxel_markers(self, stamp):
        """Visualize the 3D voxel map in RViz as cube markers (alternative)."""
        if not self.voxel_map:
            return

        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = "voxels"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = self.voxel_resolution
        marker.scale.y = self.voxel_resolution
        marker.scale.z = self.voxel_resolution

        for (vx, vy, vz), occupied in self.voxel_map.items():
            if not occupied:
                continue
            p = Point()
            p.x = float(vx) * self.voxel_resolution
            p.y = float(vy) * self.voxel_resolution
            p.z = float(vz) * self.voxel_resolution
            marker.points.append(p)

            height_norm = min(max(p.z / 2.0, 0.0), 1.0)
            c = ColorRGBA()
            c.r = height_norm
            c.g = 0.3
            c.b = 1.0 - height_norm
            c.a = 0.7
            marker.colors.append(c)

        marker_array.markers.append(marker)
        self.voxel_pub.publish(marker_array)

    def _publish_frontier_markers(self, stamp):
        """Visualize current frontier cells as green spheres in RViz."""
        grid = self.occupancy_grid.copy()
        free_mask = (grid == 0)
        unknown_mask = (grid == -1)

        kernel = np.ones((3, 3), dtype=np.uint8)
        unknown_dilated = cv2.dilate(
            unknown_mask.astype(np.uint8), kernel, iterations=1
        )
        frontier_mask = free_mask & (unknown_dilated > 0)

        marker_array = MarkerArray()

        # Delete previous frontier markers
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.header.stamp = stamp
        delete_marker.ns = "frontiers"
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        frontier_cells = np.argwhere(frontier_mask)
        if len(frontier_cells) == 0:
            self.frontier_pub.publish(marker_array)
            return

        # Display frontier as a point list (lightweight)
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = "frontiers"
        marker.id = 1
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = self.map_resolution
        marker.scale.y = self.map_resolution
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.5, a=0.8)

        # Subsample for performance if too many frontier cells
        step = max(1, len(frontier_cells) // 500)
        for row, col in frontier_cells[::step]:
            p = Point()
            p.x = float(col) * self.map_resolution + self.map_origin_x
            p.y = float(row) * self.map_resolution + self.map_origin_y
            p.z = 0.05
            marker.points.append(p)

        marker_array.markers.append(marker)
        self.frontier_pub.publish(marker_array)

    def _set_state(self, new_state: ExplorationState):
        self.state = new_state
        self.get_logger().info(f"State → {new_state.value}")


def main(args=None):
    rclpy.init(args=args)
    node = MappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
