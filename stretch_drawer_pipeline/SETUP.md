# Setup

## System Requirements

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10+
- CUDA-capable GPU (for Detic + CLIP + GNN inference)

## ROS2 Dependencies

Install via rosdep:
```bash
cd ~/ament_ws
rosdep install --from-paths src --ignore-src -r -y
```

Required ROS2 packages:
- `nav2_msgs`
- `tf2_ros`
- `tf2_geometry_msgs`
- `cv_bridge`
- `sensor_msgs`
- `visualization_msgs`
- `trajectory_msgs`

## Python Dependencies

```bash
pip install numpy opencv-python scipy scikit-learn torch
pip install tf-transformations
pip install clip-by-openai   # or install from github.com/openai/CLIP
```

For Detic detection (Node 2 and Node 4):
```bash
# Detic is expected at semantic-object-container-room/realrobot/Detic/
pip install detectron2
```

Simple exploration mode (Node 1) has no additional Python dependencies beyond the ROS2 packages listed above.

## Building

```bash
cd ~/ament_ws
colcon build --packages-select stretch_drawer_pipeline --symlink-install
source install/setup.bash
```

Since this uses `--symlink-install`, Python file edits do not require a rebuild. However, adding or modifying `.srv`/`.msg` files requires a full rebuild.

## Simulation Setup (MuJoCo)

1. Ensure `stretch_simulation` package is built and the MuJoCo driver launches:
   ```bash
   ros2 launch stretch_simulation stretch_mujoco_driver.launch.py \
     use_cameras:=true use_rviz:=false mode:=navigation
   ```

2. Verify camera topics:
   ```bash
   ros2 topic list | grep camera
   # Should see /camera/color/image_raw, /camera/depth/image_rect_raw, /camera/color/camera_info
   ```

3. Launch the pipeline:
   ```bash
   ros2 launch stretch_drawer_pipeline pipeline_with_gnn.launch.py use_sim:=true
   ```

## Real Robot Setup

The real robot uses a split architecture:
- **Robot**: runs `stretch_driver`, publishes camera and TF
- **Workstation (automata-3)**: runs Node 2 + Node 4 (Detic + GNN on GPU) via rosbridge

### On the robot:

1. Launch the Stretch driver:
   ```bash
   ros2 launch stretch_core stretch_driver.launch.py
   ```

2. Verify TF tree:
   ```bash
   ros2 run tf2_tools view_frames
   # Should see odom → base_link → ... → camera_color_optical_frame
   ```

3. Ensure the rosbridge server is running for automata-3 to connect:
   ```bash
   ros2 launch rosbridge_server rosbridge_websocket_launch.xml
   ```

### On automata-3:

1. Set the `robot_ip` parameter in config or launch args (needed by Node 2 and Node 4 for rosbridge)
2. Launch the pipeline:
   ```bash
   ros2 launch stretch_drawer_pipeline pipeline_with_gnn.launch.py
   ```

### For simple exploration mode:

No additional setup needed — simple mode uses ROS2 FollowJointTrajectory directly via the stretch driver.

```bash
ros2 launch stretch_drawer_pipeline pipeline_with_gnn.launch.py \
  exploration_mode:=simple
```

## Verifying Installation

```bash
# Check custom message/service types
ros2 interface show stretch_drawer_pipeline/msg/DrawerDetection
ros2 interface show stretch_drawer_pipeline/srv/SetRankings

# Check nodes can be started
ros2 run stretch_drawer_pipeline exploration_node.py --ros-args -p use_sim:=true
ros2 run stretch_drawer_pipeline scene_graph_node.py --ros-args -p use_sim:=true
```
