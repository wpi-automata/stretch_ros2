# Node 2: Drawer Detection

## Overview

Continuously detects drawers in camera frames using Detic, localizes handles, projects to world coordinates, and maintains a de-duplicated global list. Drawers are visualized in RViz with reachability coloring.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detection_confidence` | `0.5` | Min Detic score for drawer class |
| `enable_dedup` | `true` | Enable de-duplication of nearby detections |
| `dedup_distance_m` | `0.1` | Distance (m) to merge detections (when dedup enabled) |
| `max_reach_height` | `1.4` | Max gripper Z (m above floor) |
| `min_reach_height` | `0.1` | Min gripper Z (m above floor) |
| `max_reach_distance` | `0.6` | Max arm extension (m) |
| `detection_rate_hz` | `2.0` | How often to run detection |
| `test_mode` | `false` | Process all frames (don't wait for exploration) |
| `rank_via_LOCUS` | `false` | Enable GNN ranking stub |
| `use_sim` | `false` | Simulation mode |

## Detection Pipeline

1. **Single-pass Detic detection**: Detic 21K vocabulary detects both drawer-class and handle-class objects in one pass on the full image
2. **Handle-to-drawer association**: Each handle is matched to a drawer by checking if the handle bbox center falls within a drawer bbox; the highest-confidence match wins
3. **Grasp point**: The center of the handle's bounding box
4. **World Projection**: Handle center pixel → depth → camera-to-map TF → world XYZ; drawer corners also projected to world for visualization
5. **Orientation**: Handle bbox aspect ratio determines horizontal vs vertical
6. **Reachability**: Z-height check against Stretch3 workspace limits
7. **De-duplication** (optional, off by default): New detection within `dedup_distance_m` of existing → merge (weighted average position). Enable with `enable_dedup: true`

## Output Format

Each drawer contains:
- `drawer_id`: Unique string identifier
- `annotated_image`: RGB with bbox overlays
- `handle_center_world`: {x, y, z} in map frame
- `handle_grasp_world`: {x, y, z} grasp point (center of handle bbox)
- `drawer_corners_world`: 4 corner points [{x, y, z}, ...] of the drawer bbox in world frame
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

- Green cube: Reachable drawer, sized to the drawer's world-frame bounding box
- Red cube: Unreachable drawer, sized to the drawer's world-frame bounding box
- Blue sphere: Handle grasp point (center of handle bbox)
- White text labels: ID, reachability, orientation, distance

## Debug

### Debug images

Each detection pass saves an annotated image to `/tmp/detic_debug/`. The image shows:
- All Detic detections in gray with class name and score labels
- Matched drawer bounding boxes in blue
- Matched handle bounding boxes in green
- Red dot at the handle grasp point (center of handle bbox)

This is useful for seeing what Detic is classifying objects as and whether handles are being matched to the correct drawers.

### Verbose logging

To see per-detection class names, scores, and bounding boxes in the console, launch with DEBUG log level:

```bash
ros2 launch stretch_drawer_pipeline test_navigate.launch.py use_sim:=true --log-level drawer_detection_node:=debug
```

At the default log level, the node prints:
- A summary of each detection pass (total detections, drawer count, handle count)
- Warnings when no handle is found inside a drawer bbox (fallback to center)
- Info when a handle is successfully matched to a drawer
