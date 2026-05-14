# Stretch Drawer Pipeline

A four-node ROS2 pipeline for Stretch3 that explores a room, detects drawers, ranks them with a GNN, and navigates to open them. Designed for both MuJoCo simulation (robocasa environments) and real-robot deployment.

## Architecture

```
┌──────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│  Node 1:         │     │  Node 2:             │     │  Node 3:           │
│  Exploration     │────▶│  Drawer Detection    │────▶│  Navigate & Open   │
│  (3 modes)       │     │                      │     │                    │
└──────────────────┘     └──────────────────────┘     └────────────────────┘
  - funmap (default)       - Detic drawer+handle       - Path planning
  - occupancy grid (TBD)   - Handle localization       - Force-based grasp
  - simple_explorer        - World projection          - Pull open
  - /exploration_status    - De-duplication             - Pressure feedback
        │                         ▲
        │                         │ /detection/set_rankings
        ▼                         │
┌──────────────────┐              │
│  Node 4:         │──────────────┘
│  Scene Graph     │
│  (GNN ranking)   │
└──────────────────┘
  - Detic all objects
  - VoxelGraphBuilder
  - ContextGNN scoring
```

## Nodes

### 1. Exploration (`exploration_node.py`)

Multi-mode exploration with pause-for-detection synchronization. At each pause point, Node 2 runs exactly one detection pass then stops. Three modes:

- **funmap**: scan-drive loop via stretch_funmap services (default)
- **occupancy_grid**: stub for future occupancy-grid frontier planner
- **simple_explorer**: lightweight structured coverage from `semantic-object-container-room`

### 2. Drawer Detection (`drawer_detection_node.py`)

Uses Detic to detect drawer and handle bounding boxes. Handles are matched to drawers by containment. Projects handle centers and drawer corners to world coordinates. De-duplicates across frames. Receives GNN rankings from Node 4 via `/detection/set_rankings` and uses spatial proximity matching to apply scores to detected drawers.

### 3. Navigate and Open (`navigate_open_node.py`)

Receives the drawer list, selects the best target (by GNN ranking, then confidence, then distance), navigates to an approach pose, orients the wrist, extends the arm until contact, closes the gripper, and pulls back using force feedback.

### 4. Scene Graph (`scene_graph_node.py`)

Runs a full-vocabulary Detic pass on all object classes (not just drawers) to build scene context. Accumulates detections in a VoxelGraphBuilder, then runs the ContextGNN to rank containers by how likely they are to hold the target object. Pushes rankings to Node 2.

## Quick Start

### Simulation

```bash
# Terminal 1: Launch MuJoCo simulation
ros2 launch stretch_simulation stretch_mujoco_driver.launch.py \
  use_cameras:=true use_rviz:=false mode:=navigation

# Terminal 2: Launch the pipeline with GNN ranking
ros2 launch stretch_drawer_pipeline pipeline_with_gnn.launch.py use_sim:=true

# Terminal 3: Start exploration
ros2 service call /mapping/start std_srvs/srv/Trigger

# After exploration completes, open the top-ranked drawer:
ros2 service call /navigate_open/execute std_srvs/srv/Trigger
```

### Real Robot

The real robot runs a split architecture: the robot runs the driver and publishes camera/TF, while a GPU workstation (automata-3) runs detection and GNN nodes via rosbridge.

```bash
# On the robot: Launch Stretch driver
ros2 launch stretch_core stretch_driver.launch.py

# On automata-3: Launch the pipeline
ros2 launch stretch_drawer_pipeline pipeline_with_gnn.launch.py

# On automata-3: Start exploration
ros2 service call /mapping/start std_srvs/srv/Trigger
```

### Using simple_explorer mode (real robot only)

```bash
ros2 launch stretch_drawer_pipeline pipeline_with_gnn.launch.py \
  exploration_mode:=simple_explorer
```

This requires `robot_ip` set in `mapping_params.yaml` and a running stretch_ros2_bridge server on the robot for ZMQ communication.

## Sim vs Real Differences

| Aspect | Simulation | Real Robot |
|--------|-----------|------------|
| **Driver** | `stretch_mujoco_driver` | `stretch_driver` |
| **Image transport** | DDS (direct ROS2 topics) | rosbridge WebSocket (compressed) |
| **Image rotation** | None (1280x720) | 90 CW (720x1280) |
| **Camera K** | Original intrinsics | Rotated intrinsics |
| **TF source** | MuJoCo driver (local) | rosbridge relay from robot |
| **Force feedback** | Unavailable (zero effort) | Real effort from `/joint_states` |
| **Pull strategy** | Fixed distance retract | Incremental with force threshold |
| **Funmap** | Launched by pipeline | Pre-launched or launched by pipeline |
| **Node 2 & 4 location** | Same machine as sim | GPU workstation (automata-3) |
| **`use_sim_time`** | `true` | `false` (wall clock) |

## Exploration Flow

```
exploration_node starts
  │
  ├─ [SCANNING] head scan or head sweep
  │
  ├─ [PAUSED_FOR_DETECTION] robot is still
  │   ├─ Node 1 calls /detection/trigger on Node 2
  │   │   └─ Node 2 runs one Detic pass (drawers+handles), then stops
  │   └─ Node 4 processes the same frame (all objects → VoxelGraphBuilder)
  │
  ├─ [DRIVING] move to next position
  │   └─ Node 2 does NOT detect (exploring=False)
  │
  └─ repeat until complete
       │
       ├─ Node 1 calls /scene_graph/build_and_rank on Node 4
       │   ├─ Node 4 runs GNN scoring
       │   └─ Node 4 calls /detection/set_rankings on Node 2
       │       └─ Node 2 matches GNN containers to drawers spatially
       │
       └─ [COMPLETE]
```

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/exploration_status` | `std_msgs/String` | Exploration state (idle/scanning/driving/paused_for_detection/complete) |
| `/drawer_markers` | `visualization_msgs/MarkerArray` | Drawer bboxes in RViz |
| `/drawer_detections_json` | `std_msgs/String` | All detected drawers (JSON) |
| `/detection/chosen_drawer_json` | `std_msgs/String` | Selected drawer for opening |
| `/navigate_open/path` | `nav_msgs/Path` | Planned path to drawer |
| `/navigate_open/status` | `std_msgs/String` | Navigation state |

## Services

| Service | Type | Provider | Description |
|---------|------|----------|-------------|
| `/mapping/start` | Trigger | Node 1 | Begin exploration |
| `/mapping/stop` | Trigger | Node 1 | Stop exploration |
| `/mapping/is_complete` | Trigger | Node 1 | Check if mapping done |
| `/detection/trigger` | Trigger | Node 2 | Force one detection pass |
| `/detection/get_drawers` | Trigger | Node 2 | Get drawer list (JSON) |
| `/detection/choose_drawer` | Trigger | Node 2 | Select best drawer |
| `/detection/set_rankings` | SetRankings | Node 2 | Inject GNN rankings |
| `/scene_graph/build_and_rank` | Trigger | Node 4 | Run GNN, push rankings |
| `/scene_graph/get_rankings` | Trigger | Node 4 | Get current rankings |
| `/scene_graph/score_now` | Trigger | Node 4 | Force re-scoring |
| `/navigate_open/execute` | Trigger | Node 3 | Navigate and open drawer |
| `/navigate_open/stop` | Trigger | Node 3 | Abort operation |

## Mode Switch Requirements

The Stretch driver has two control modes:

| Mode | Service | What works |
|------|---------|------------|
| **Navigation** (default) | `/switch_to_navigation_mode` | `cmd_vel` velocity commands |
| **Position** | `/switch_to_position_mode` | `translate_mobile_base`, `rotate_mobile_base` via FollowJointTrajectory |

Node 1 runs in navigation mode. Node 3 switches to position mode for drawer manipulation and back to navigation mode in a `finally` block.

## See Also

- [NODE1_MAPPING.md](stretch_drawer_pipeline/NODE1_MAPPING.md) — Exploration node details
- [NODE2_DETECTION.md](stretch_drawer_pipeline/NODE2_DETECTION.md) — Detection node details
- [NODE3_NAVIGATE.md](stretch_drawer_pipeline/NODE3_NAVIGATE.md) — Navigate & open details
- [NODE4_SCENE_GRAPH.md](stretch_drawer_pipeline/NODE4_SCENE_GRAPH.md) — Scene graph & GNN details
- [SETUP.md](SETUP.md) — Dependencies and installation
- [TESTING.md](TESTING.md) — How to test each node independently
