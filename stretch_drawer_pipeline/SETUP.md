# Setup

## System Requirements

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10+
- CUDA-capable GPU (for Detic inference; CPU fallback available)

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
```

For Detic detection (Node 2):
```bash
# Detic is expected at semantic-object-container-room/realrobot/Detic/
# Follow the Detic setup instructions in that directory
pip install detectron2
```

## Building

```bash
cd ~/ament_ws
colcon build --packages-select stretch_drawer_pipeline --symlink-install
source install/setup.bash
```

Since this uses `--symlink-install`, Python file edits do not require a rebuild.

## Simulation Setup (MuJoCo)

1. Ensure `stretch_simulation` package is built and the MuJoCo driver launches correctly:
   ```bash
   ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_robocasa:=true
   ```

2. Verify camera topics are publishing:
   ```bash
   ros2 topic list | grep camera
   ```

## Real Robot Setup

1. Ensure the Stretch3 driver is running:
   ```bash
   ros2 launch stretch_core stretch_driver.launch.py
   ```

2. Verify TF tree is publishing (map → odom → base_link):
   ```bash
   ros2 run tf2_tools view_frames
   ```

3. Ensure Nav2 stack is running for goal-pose navigation:
   ```bash
   ros2 launch stretch_nav2 navigation.launch.py
   ```

## Verifying Installation

```bash
# Check that message types are available
ros2 interface show stretch_drawer_pipeline/msg/DrawerDetection
ros2 interface show stretch_drawer_pipeline/srv/NavigateToDrawer

# Check nodes can be started
ros2 run stretch_drawer_pipeline mapping_node.py --ros-args -p use_sim:=true
```
