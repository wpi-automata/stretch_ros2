# Node 1: Exploration (`exploration_node.py`)

## Overview

Multi-mode exploration that replaces the old single-mode `mapping_node.py`. Supports three backends: funmap, occupancy grid (stub), and simple_explorer. All modes publish `/exploration_status` and pause for detection at natural breakpoints.

## Exploration Modes

### Mode 1: `funmap` (default)

Scan-drive loop via stretch_funmap services. After each head scan, the node pauses for detection before driving to the next position.

```
head_scan → PAUSED_FOR_DETECTION → /detection/trigger → DRIVING → drive_to_scan → repeat
```

Requires funmap to be running (launched automatically in sim, or pre-launched on real).

### Mode 2: `occupancy_grid` (stub)

Placeholder for future occupancy-grid frontier planner. Currently logs a warning and sets state to COMPLETE.

### Mode 3: `simple_explorer`

Imports `SimpleExplorer` from `semantic-object-container-room/realrobot/stretch/explore_simple.py`. Connects to the robot via ZMQ (`HomeRobotZmqClient`) and executes a structured coverage pattern:

1. Rotate 360 at start position with head sweeps (8 angles per rotation)
2. Move to new positions (forward, try alternate directions if blocked)
3. Full 360 sweep at each new position
4. At each head angle: `PAUSED_FOR_DETECTION` → call `/detection/trigger` → continue

Requires `robot_ip` parameter and a running stretch_ros2_bridge server.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exploration_mode` | `"funmap"` | `funmap`, `occupancy_grid`, or `simple_explorer` |
| `exploration_timeout_s` | `300.0` | Max exploration time (funmap mode) |
| `max_scan_drive_cycles` | `20` | Max cycles (funmap mode) |
| `use_sim` | `false` | Simulation mode |
| `robot_ip` | `""` | Robot IP for simple_explorer ZMQ |
| `n_positions` | `4` | Positions to visit (simple_explorer mode) |
| `room_type` | `"kitchen"` | Scene context for GNN |
| `query` | `""` | Target object for GNN ranking |

## Exploration States

| State | Description |
|-------|-------------|
| `idle` | Not exploring |
| `scanning` | Head scan or head sweep in progress |
| `driving` | Moving to next position |
| `paused_for_detection` | Robot is still — Node 2 should detect |
| `complete` | Exploration finished |

## Integration with Node 2

Detection is gated by state. Node 2 only detects when:
1. The exploration status is `paused_for_detection`, OR
2. `/detection/trigger` is called explicitly

After each trigger call, Node 2 runs exactly one detection pass and sets `exploring=False`, so it does not detect while the robot is moving.

## Integration with Node 4

When exploration completes, Node 1 calls `/scene_graph/build_and_rank` on Node 4, which:
1. Clusters accumulated detections with DBSCAN
2. Builds a HeteroData scene graph
3. Runs the ContextGNN to score containers
4. Pushes rankings to Node 2 via `/detection/set_rankings`

## Services

| Service | Type | Description |
|---------|------|-------------|
| `/mapping/start` | Trigger | Begin exploration |
| `/mapping/stop` | Trigger | Stop exploration early |
| `/mapping/is_complete` | Trigger | Query completion status |

## Publications

| Topic | Type | Description |
|-------|------|-------------|
| `/exploration_status` | `std_msgs/String` | Current exploration state |

## Sim vs Real

| Aspect | Simulation | Real Robot |
|--------|-----------|------------|
| Funmap | Launched by pipeline | Pre-launched or launched by pipeline |
| simple_explorer | Not typically used | Primary use case (ZMQ to robot) |
| `use_sim_time` | `true` | `false` |
