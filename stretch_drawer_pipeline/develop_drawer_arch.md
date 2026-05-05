# Development Log: stretch_drawer_pipeline

## Architecture Decisions

### Package Structure
- `ament_cmake` build type (not `ament_python`) because we need `rosidl_generate_interfaces` for custom messages
- Three independent ROS2 nodes that communicate via topics/services
- Single RViz config (`rviz/mapping.rviz`) used by all launch files

### Simulation Integration
- **Launch sim driver SEPARATELY** in its own terminal:
  `ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_cameras:=true use_rviz:=false mode:=navigation`
- Pipeline and test launch files do NOT include the sim driver
- Reason: sim driver's launch file uses `sys.argv` to detect robocasa layout/style args — `IncludeLaunchDescription` doesn't populate `sys.argv`, so interactive prompts always fire and double-launching occurs
- Sim driver provides: robot URDF, joint_state_publisher, robot_state_publisher, TF (map→odom→base_link), cameras, lidar

### TF Tree (from sim driver)
- `map` → `odom` (static identity transform, broadcast by sim driver line 385-387)
- `odom` → `base_link` (dynamic, broadcast by sim driver)
- `base_link` → all joints (from robot_state_publisher + joint_state_publisher)

### Sim Driver Topics Used
- `/camera/depth/color/points` (PointCloud2, frame: `camera_depth_optical_frame`, QoS: **BEST_EFFORT**)
- `/camera/color/image_raw` (Image, frame: `camera_color_optical_frame`, QoS: BEST_EFFORT)
- `/camera/depth/image_rect_raw` (Image, depth)
- `/stretch/joint_states` (JointState)
- `/scan` (LaserScan, QoS: BEST_EFFORT)
- `/stretch/cmd_vel` (Twist, for navigation mode)

### Critical QoS Fix
The sim driver publishes sensor data with `BEST_EFFORT` reliability. Subscribers MUST use `BEST_EFFORT` QoS or they'll never receive messages. This was the root cause of the occupancy grid not publishing.

### Node Communication Flow
```
Node 1 (Mapping) → publishes /exploration_status, /map
Node 2 (Detection) → subscribes /exploration_status, /camera topics
                    → publishes /drawer_markers, serves /detection/get_drawers
Node 3 (Navigate)  → calls /detection/get_drawers service
                    → publishes /navigate_open/path, /navigate_open/target_marker
```

## Files Created

```
stretch_drawer_pipeline/
├── CMakeLists.txt
├── package.xml
├── README.md
├── SETUP.md
├── TESTING.md
├── EVAL.md
├── OPEN_DRAWERS_SEMANTICS.md
├── config/
│   ├── mapping_params.yaml
│   ├── detection_params.yaml
│   └── navigate_params.yaml
├── launch/
│   ├── pipeline_sim.launch.py      (full pipeline + mujoco + rviz)
│   ├── pipeline_real.launch.py     (full pipeline, no sim)
│   ├── test_mapping.launch.py      (mapping only + mujoco + rviz)
│   ├── test_detection.launch.py    (detection test + mujoco + rviz)
│   └── test_navigate.launch.py     (detect+nav test + mujoco + rviz)
├── msg/
│   ├── DrawerDetection.msg
│   └── DrawerList.msg
├── srv/
│   ├── NavigateToDrawer.srv
│   └── TriggerDetection.srv
├── resource/
│   └── stretch_drawer_pipeline
├── rviz/
│   └── mapping.rviz
└── stretch_drawer_pipeline/
    ├── __init__.py
    ├── mapping_node.py
    ├── drawer_detection_node.py
    ├── navigate_open_node.py
    ├── NODE1_MAPPING.md
    ├── NODE2_DETECTION.md
    └── NODE3_NAVIGATE.md
```

## Issues Found & Fixed

1. **QoS mismatch**: Mapping node subscribed with RELIABLE, sim publishes BEST_EFFORT → no data received → empty occupancy grid. Fixed by using BEST_EFFORT QoS on the subscriber.

2. **Double MuJoCo launch**: Early design had test launches include sim driver AND user might launch sim separately. Fixed: pipeline_sim.launch.py is the single entry point that includes the sim driver with `use_rviz:=false` (uses our own rviz config instead).

3. **Build directory conflict**: First build failed due to stale `build/` directory from a previous attempt. Fixed by `rm -rf build/stretch_drawer_pipeline install/stretch_drawer_pipeline`.

## Pending Work

- Detection node camera subscriber also needs BEST_EFFORT QoS
- Verify end-to-end data flow in simulation
- Test that goal_pose commands work with the sim driver in navigation mode
- Verify the occupancy grid actually fills with data from the pointcloud
