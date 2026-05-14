# Node 1: Exploration (`exploration_node.py`)

## Overview

Multi-mode exploration that replaces the old single-mode `mapping_node.py`. Supports three backends: funmap, occupancy grid (stub), and simple. All modes publish `/exploration_status` and pause for detection at natural breakpoints.

This node runs on the robot itself (not automata-3) and controls the base and head via ROS2 FollowJointTrajectory.

## Exploration Modes

### Mode 1: `funmap` (default)

Scan-drive loop via stretch_funmap services. After each head scan, the node pauses for detection before driving to the next position.

```
head_scan -> PAUSED_FOR_DETECTION -> /detection/trigger -> DRIVING -> drive_to_scan -> repeat
```

Requires funmap to be running (launched automatically in sim, or pre-launched on real).

### Mode 2: `occupancy_grid` (stub)

Placeholder for future occupancy-grid frontier planner. Currently logs a warning and sets state to COMPLETE.

### Mode 3: `simple`

Lightweight structured coverage using direct ROS2 joint commands (FollowJointTrajectory). No external dependencies beyond the stretch driver.

1. Switches driver to position mode
2. At each position: rotates 360 in 4 steps (90 each), performing a head sweep at each rotation step
3. Head sweep: 8 angles (4 pan x 2 tilt), pauses for detection at each angle
4. After each full rotation, moves forward (`move_distance` meters) with stall detection
5. Repeats for `n_positions` positions total
6. On completion: calls `/scene_graph/build_and_rank`, then sets COMPLETE

Movement uses TF-based stall detection: if the robot moves less than 5cm during a translate command, it tries rotating 90 to find an alternate direction.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exploration_mode` | `"funmap"` | `funmap`, `occupancy_grid`, or `simple` |
| `exploration_timeout_s` | `300.0` | Max exploration time (funmap mode) |
| `max_scan_drive_cycles` | `20` | Max cycles (funmap mode) |
| `use_sim` | `false` | Simulation mode |
| `n_positions` | `4` | Positions to visit (simple mode) |
| `move_distance` | `0.8` | Meters to move between positions (simple mode) |
| `room_type` | `"kitchen"` | Scene context for GNN |
| `query` | `""` | Target object for GNN ranking |

## Exploration States

| State | Description |
|-------|-------------|
| `idle` | Not exploring |
| `scanning` | Head scan or head sweep in progress |
| `driving` | Moving to next position |
| `paused_for_detection` | Robot is still -- Node 2 should detect |
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
| Simple mode | Works (MuJoCo FollowJointTrajectory) | Primary use case |
| Runs on | Same machine as sim | On the robot (not automata-3) |
| `use_sim_time` | `true` | `false` |
