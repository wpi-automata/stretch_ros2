# Node 2: Drawer Detection

## Overview

Continuously detects drawers in camera frames using Detic, localizes handles, projects to world coordinates, and maintains a de-duplicated global list. Drawers are visualized in RViz with reachability coloring.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detection_confidence` | `0.5` | Min Detic score for drawer class |
| `dedup_distance_m` | `0.3` | Distance (m) to merge detections |
| `max_reach_height` | `1.4` | Max gripper Z (m above floor) |
| `min_reach_height` | `0.1` | Min gripper Z (m above floor) |
| `max_reach_distance` | `0.6` | Max arm extension (m) |
| `detection_rate_hz` | `2.0` | How often to run detection |
| `test_mode` | `false` | Process all frames (don't wait for exploration) |
| `rank_via_LOCUS` | `false` | Enable GNN ranking stub |
| `use_sim` | `false` | Simulation mode |

## Detection Pipeline

1. **Drawer Detection**: Detic 21K vocabulary detects drawer-class objects
2. **Handle Finding**: Second Detic pass within the drawer bbox for handle/knob classes
3. **World Projection**: Handle center pixel → depth → camera-to-map TF → world XYZ
4. **Orientation**: Handle bbox aspect ratio determines horizontal vs vertical
5. **Reachability**: Z-height check against Stretch3 workspace limits
6. **De-duplication**: New detection within `dedup_distance_m` of existing → merge (weighted average position)

## Output Format

Each drawer contains:
- `drawer_id`: Unique string identifier
- `annotated_image`: RGB with bbox overlays
- `handle_center_world`: {x, y, z} in map frame
- `handle_grasp_world`: {x, y, z} grasp point
- `reachable`: boolean
- `handle_orientation`: "horizontal" or "vertical"
- `ranking`: float (0.0 if TBD)
- `distance_to_robot`: float meters

## LOCUS GNN Ranking (Stub)

When `--rank_via_LOCUS` is set, the node calls a stub that documents what data would be needed for the GNN ranking:
- Drawer CLIP embeddings
- Drawer world positions
- Nearby object types/positions
- Room type
- Spatial edges between containers
- Target query embedding

## RViz Visualization

- Green boxes: Reachable drawers
- Red boxes: Unreachable drawers
- Text labels: ID, reachability, orientation, distance
