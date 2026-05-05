# Node 1: Mapping and Exploration

## Overview

Frontier-based autonomous exploration that builds both a 3D voxel map and a 2D occupancy grid. The robot rotates to seed the map, then iteratively navigates to frontier cells until the room is fully explored.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frontier_method` | `"occupancy_grid"` | `"occupancy_grid"` or `"voxel"` |
| `map_resolution` | `0.05` | Meters per grid cell |
| `map_width` | `200` | Grid width in cells |
| `map_height` | `200` | Grid height in cells |
| `voxel_resolution` | `0.05` | Meters per voxel |
| `exploration_timeout_s` | `300.0` | Max exploration time |
| `min_frontier_size` | `5` | Min cells for valid frontier |
| `robot_radius` | `0.18` | Robot collision radius |
| `max_exploration_attempts` | `50` | Max navigation attempts |
| `use_sim` | `false` | Simulation mode |

## Algorithm

1. **Initialization**: Rotate 360 degrees to seed the occupancy grid
2. **Frontier Detection**: Find free cells adjacent to unknown cells, cluster them
3. **Scoring**: Rank frontiers by (size / distance) — prefer large, nearby frontiers
4. **Navigation**: Send goal_pose to Nav2 and wait for arrival
5. **Repeat**: From new position, detect new frontiers until none remain

## Integration with Node 2

While this node explores, Node 2's detection runs on every incoming camera frame. The `/exploration_status` topic tells Node 2 whether exploration is active, so it can time its processing appropriately.

## Subscriptions

- `/camera/depth/color/points` — Point cloud for map integration

## Publications

- `/map` — 2D occupancy grid (latched, for Nav2 and RViz)
- `/voxel_markers` — 3D cube markers for RViz
- `/exploration_status` — String: idle, rotating, planning, navigating, complete
- `/stretch/cmd_vel` — Rotation commands
- `/goal_pose` — Navigation goals
