# Open Drawers — Semantic Pipeline

## New Functionality: `stretch_drawer_pipeline`

A three-node ROS2 pipeline that autonomously maps a room, detects all drawers with their handles, and opens the best-ranked reachable drawer.

### Nodes

1. **Mapping & Exploration** (`mapping_node.py`)
   - Frontier-based room exploration
   - Builds 2D occupancy grid + 3D voxel map simultaneously
   - Configurable frontier method (2D occupancy grid or 3D voxel)
   - Publishes `/map` for Nav2 and RViz, `/voxel_markers` for 3D visualization

2. **Drawer Detection** (`drawer_detection_node.py`)
   - Detic 21K detection for drawer bounding boxes
   - Handle localization within each drawer bbox
   - World-frame projection via depth + camera TF
   - De-duplication using distance threshold (from `semantic-object-container-room`)
   - Reachability check based on Stretch3 kinematic workspace
   - Output: list of `DrawerDetection` messages with handle positions, orientations, reachability
   - Stub for LOCUS GNN ranking (`--rank_via_LOCUS`)

3. **Navigate & Open** (`navigate_open_node.py`)
   - Selects best drawer (ranking > distance)
   - Switches driver to position mode for base translate/rotate commands
   - Navigates base to approach pose (0.45m away)
   - Aligns perpendicular to drawer face using corner geometry normal
   - Wrist orientation matches handle type (horizontal/vertical)
   - Force-feedback approach until contact
   - Gripper close, pull-back until force threshold
   - Release and retract
   - Switches driver back to navigation mode (in `finally` block)

### Integration with `semantic-object-container-room`

- Uses `realrobot.detector.VisualDetector` and `nms_by_type` for Detic detection
- Uses `realrobot.stretch.projection.project_bbox_to_world_se3` for 3D projection
- Uses DBSCAN-based de-duplication pattern from `realrobot.voxel_graph_builder`
- Stub prepared for `realrobot.inference.score_containers` GNN ranking

### Launch Files

| File | Purpose |
|------|---------|
| `pipeline_sim.launch.py` | Full pipeline in MuJoCo simulation |
| `pipeline_real.launch.py` | Full pipeline on real Stretch3 |
| `test_mapping.launch.py` | Independent mapping test |
| `test_detection.launch.py` | Independent detection test (spiral pattern) |
| `test_navigate.launch.py` | Interactive navigation test (user drives + triggers) |

### Mode Switching

The Stretch driver has two control modes. The pipeline switches between them automatically:
- **Navigation mode** (default): `cmd_vel` works, base translate/rotate via FollowJointTrajectory ignored. Used by Node 1 (frontier exploration).
- **Position mode**: base translate/rotate via FollowJointTrajectory works, `cmd_vel` ignored. Used by Node 3 (navigate to drawer).

Node 3 calls `/switch_to_position_mode` before base movement and `/switch_to_navigation_mode` in a `finally` block when done. Both the MuJoCo sim driver and real Stretch driver expose these services.

### Sim2Real

The pipeline is designed for sim2real transfer:
- Same ROS2 topic interface in both environments
- Force feedback works with MuJoCo simulated sensors and real Stretch3 hardware
- Camera topics and TF tree are identical between sim and real
- `use_sim` parameter toggles simulation-specific behavior where needed
- Mode switching services (`/switch_to_position_mode`, `/switch_to_navigation_mode`) exist on both sim and real drivers
