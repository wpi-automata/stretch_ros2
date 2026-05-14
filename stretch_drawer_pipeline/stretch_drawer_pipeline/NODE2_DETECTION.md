# Node 2: Drawer Detection (`drawer_detection_node.py`)

## Overview

Detects drawers in camera frames using Detic, localizes handles, projects to world coordinates, and maintains a de-duplicated global list. Receives GNN rankings from Node 4 and applies them to drawers via spatial proximity matching.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detection_confidence` | `0.5` | Min Detic score for drawer class |
| `enable_dedup` | `true` | Enable de-duplication of nearby detections |
| `dedup_distance_m` | `0.1` | Distance (m) to merge detections |
| `max_reach_height` | `1.4` | Max gripper Z (m above floor) |
| `min_reach_height` | `0.1` | Min gripper Z (m above floor) |
| `max_reach_distance` | `0.6` | Max arm extension (m) |
| `detection_rate_hz` | `2.0` | How often to run detection |
| `test_mode` | `false` | Process all frames (don't wait for exploration) |
| `gnn_match_threshold` | `0.5` | Max distance (m) to match GNN container to drawer |
| `use_sim` | `false` | Simulation mode |

## Detection Pipeline

1. **Detic detection**: Single pass detects drawer-class and handle-class objects
2. **Handle-to-drawer association**: Handle bbox center within drawer bbox
3. **World projection**: Handle center pixel → depth → camera-to-odom TF → world XYZ
4. **Orientation**: Handle bbox aspect ratio → horizontal or vertical
5. **Reachability**: Z-height check against Stretch3 workspace limits
6. **De-duplication**: New detection within `dedup_distance_m` of existing → merge

## Detection Gating

Detection does NOT run continuously. It is gated by the exploration node:

- **`paused_for_detection`** state → `exploring=True` (timer can detect)
- All other states → `exploring=False` (timer skips)
- `/detection/trigger` service → runs one pass, then sets `exploring=False`
- `test_mode=true` → bypasses gating, detects every frame

This ensures detection only runs when the robot is stationary and the camera is stable.

## GNN Ranking Integration

Node 4 (scene_graph_node) pushes rankings via `/detection/set_rankings`. The matching algorithm:

1. For each GNN container whose `container_type` is in the drawer class set:
2. Find all Node 2 drawers within `gnn_match_threshold` (default 0.5m)
3. Each drawer matches the closest GNN container within threshold
4. One GNN container can match multiple drawers (e.g. dresser with several drawers)
5. Unmatched drawers keep `ranking = 0.0`

Drawer selection (`/detection/choose_drawer`) sorts by:
```
(-ranking, -confidence, distance_to_robot)
```

## Output Format

Each drawer contains:
- `drawer_id`: Unique string identifier
- `handle_center_world`: {x, y, z} in odom frame
- `handle_grasp_world`: {x, y, z} grasp point
- `drawer_corners_world`: 4 corner points for 3D visualization
- `reachable`: boolean
- `handle_orientation`: "horizontal" or "vertical"
- `ranking`: float (GNN score, 0.0 if unmatched)
- `confidence`: float (Detic detection score)
- `distance_to_robot`: float meters

## Sim vs Real

| Aspect | Simulation | Real Robot |
|--------|-----------|------------|
| Image transport | DDS (raw topics) | rosbridge WebSocket (compressed) |
| Image rotation | None (1280x720) | 90 CW → 720x1280 |
| Camera K | Original D435i intrinsics | Rotated intrinsics (fx/fy swapped, cx/cy adjusted) |
| TF source | Local TF buffer (DDS) | Rosbridge relay → local republish |
| Runs on | Same machine as sim | GPU workstation (automata-3) |

## Services

| Service | Type | Description |
|---------|------|-------------|
| `/detection/trigger` | Trigger | Run one detection pass, then stop |
| `/detection/get_drawers` | Trigger | Get drawer list as JSON |
| `/detection/choose_drawer` | Trigger | Select best drawer by ranking |
| `/detection/set_rankings` | SetRankings | Receive GNN rankings from Node 4 |

## Debug

Debug images are saved to `/tmp/detic_debug/` on each detection pass. All GNN matching decisions are logged with drawer IDs, GNN container IDs, distances, and scores.
