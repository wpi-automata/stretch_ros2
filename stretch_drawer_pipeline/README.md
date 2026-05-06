# Stretch Drawer Pipeline

A three-node ROS2 pipeline for Stretch3 that maps a room, detects drawers, and navigates to open them. Designed for both MuJoCo simulation (robocasa environments) and real-robot deployment.

## Architecture

```
┌─────────────────┐     ┌──────────────��───────┐     ┌────────────────────┐
│  Node 1:        │     │  Node 2:             │     │  Node 3:           │
│  Mapping &      │────▶│  Drawer Detection    │────▶│  Navigate & Open   │
│  Exploration    │     │                      │     │                    │
└─────────────────┘     └──────────────────────┘     └────────────────────┘
  - Frontier nav          - Detic detection           - Path planning
  - 2D occ grid           - Handle localization       - Force-based grasp
  - 3D voxel map          - World projection          - Pull open
  - /map publish           - De-duplication            - Pressure feedback
```

## Nodes

### 1. Mapping and Exploration (`mapping_node.py`)

Explores the room using frontier-based exploration. Builds both a 3D voxel map and a 2D occupancy grid. Frontier selection method is configurable (2D occupancy grid default, or 3D voxel).

### 2. Drawer Detection (`drawer_detection_node.py`)

Uses Detic (from `semantic-object-container-room`) to detect drawer and handle bounding boxes in a single pass. Handles are associated with drawers by containment (handle bbox center within drawer bbox). The grasp point is the center of the handle's bounding box. Projects handle centers and drawer corners to world coordinates. De-duplicates across frames. Determines reachability based on Stretch3 workspace.

### 3. Navigate and Open (`navigate_open_node.py`)

Receives the drawer list, selects the best target (by ranking or distance), navigates to an approach pose, orients the wrist, extends the arm until contact, closes the gripper, and pulls back using force feedback.

## Quick Start

### Simulation
```bash
# Terminal 1: Launch MuJoCo simulation
ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_cameras:=true use_rviz:=true mode:=navigation

# Optional: use robocasa_seed to control which fixture the robot spawns near (default is random)
ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_cameras:=true use_rviz:=true mode:=navigation robocasa_seed:=0

# Terminal 2: Launch the full pipeline
ros2 launch stretch_drawer_pipeline pipeline_sim.launch.py

# Terminal 3: Start exploration
ros2 service call /mapping/start std_srvs/srv/Trigger
```

### Real Robot
```bash
# Terminal 1: Launch Stretch driver
ros2 launch stretch_core stretch_driver.launch.py

# Terminal 2: Launch the full pipeline
ros2 launch stretch_drawer_pipeline pipeline_real.launch.py

# Terminal 3: Start exploration
ros2 service call /mapping/start std_srvs/srv/Trigger
```

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/map` | `nav_msgs/OccupancyGrid` | 2D occupancy grid |
| `/voxel_markers` | `visualization_msgs/MarkerArray` | 3D voxel visualization |
| `/exploration_status` | `std_msgs/String` | Current exploration state |
| `/drawer_markers` | `visualization_msgs/MarkerArray` | Drawer bboxes in RViz |
| `/navigate_open/path` | `nav_msgs/Path` | Planned path to drawer |
| `/navigate_open/target_marker` | `visualization_msgs/Marker` | Pink target marker |
| `/navigate_open/status` | `std_msgs/String` | Navigation state |

## Services

| Service | Type | Description |
|---------|------|-------------|
| `/mapping/start` | `std_srvs/Trigger` | Begin frontier exploration |
| `/mapping/stop` | `std_srvs/Trigger` | Stop exploration |
| `/mapping/is_complete` | `std_srvs/Trigger` | Check if mapping done |
| `/detection/trigger` | `std_srvs/Trigger` | Force detection pass |
| `/detection/get_drawers` | `std_srvs/Trigger` | Get drawer list (JSON) |
| `/navigate_open/execute` | `std_srvs/Trigger` | Navigate to and open drawer |
| `/navigate_open/stop` | `std_srvs/Trigger` | Abort operation |

## Mode Switch Requirements

The Stretch driver (both MuJoCo sim and real robot) has two control modes:

| Mode | Service | What works | What is ignored |
|------|---------|------------|-----------------|
| **Navigation** (default) | `/switch_to_navigation_mode` | `cmd_vel` velocity commands | `translate_mobile_base`, `rotate_mobile_base` via FollowJointTrajectory |
| **Position** | `/switch_to_position_mode` | `translate_mobile_base`, `rotate_mobile_base` via FollowJointTrajectory | `cmd_vel` velocity commands |

### Where mode switches occur

1. **Node 1 (Mapping/Exploration)** runs in **navigation mode** (the default). Frontier exploration uses `cmd_vel` to drive the base.

2. **Node 3 (Navigate & Open)** switches to **position mode** before moving the base toward the drawer. It uses `translate_mobile_base` and `rotate_mobile_base` joints via the FollowJointTrajectory action server, which require position mode.

3. **Node 3** switches back to **navigation mode** in a `finally` block when the pipeline completes (success, failure, or exception). This ensures frontier exploration can resume with `cmd_vel`.

### Timeline

```
Exploration (nav mode)
  │
  ▼
Node 3 triggered
  │── switch_to_position_mode
  │── rotate toward drawer
  │── translate to approach point
  │── align perpendicular to drawer face
  │── extend arm, grasp, pull, release
  │── switch_to_navigation_mode (finally)
  ▼
Exploration can resume (nav mode)
```

### Debugging

If the robot accepts `translate_mobile_base` goals but doesn't move, it is likely still in navigation mode. Verify and fix manually:

```bash
# Check by sending a small translate and watching if the robot moves
ros2 service call /switch_to_position_mode std_srvs/srv/Trigger
ros2 action send_goal /stretch_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{trajectory: {joint_names: [translate_mobile_base], points: [{positions: [0.1], time_from_start: {sec: 2}}]}}"
```

## See Also

- [SETUP.md](SETUP.md) — Dependencies and installation
- [TESTING.md](TESTING.md) — How to test each node independently
- [EVAL.md](EVAL.md) — Full pipeline evaluation procedure
