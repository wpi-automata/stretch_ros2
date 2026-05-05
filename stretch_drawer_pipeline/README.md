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

Uses Detic (from `semantic-object-container-room`) to detect drawer bounding boxes, then localizes the handle within each bbox. Projects handle centers to world coordinates. De-duplicates across frames. Determines reachability based on Stretch3 workspace.

### 3. Navigate and Open (`navigate_open_node.py`)

Receives the drawer list, selects the best target (by ranking or distance), navigates to an approach pose, orients the wrist, extends the arm until contact, closes the gripper, and pulls back using force feedback.

## Quick Start

### Simulation
```bash
# Terminal 1: Launch MuJoCo simulation
ros2 launch stretch_simulation stretch_mujoco_driver.launch.py

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

## See Also

- [SETUP.md](SETUP.md) — Dependencies and installation
- [TESTING.md](TESTING.md) — How to test each node independently
- [EVAL.md](EVAL.md) — Full pipeline evaluation procedure
