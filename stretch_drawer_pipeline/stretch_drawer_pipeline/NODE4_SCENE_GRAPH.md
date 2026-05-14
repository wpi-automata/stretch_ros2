# Node 4: Scene Graph (`scene_graph_node.py`)

## Overview

Accumulates all Detic detections (not just drawers) during exploration, builds a scene graph, and runs the ContextGNN to rank containers by how likely they are to hold the target object. Pushes rankings to Node 2.

This node runs its own Detic pass with the full 21K vocabulary, separate from Node 2's drawer-only detection. The GNN needs scene context (landmarks like countertops, appliances, furniture) that Node 2 doesn't detect.

## How It Works

1. **During exploration**: Listens to `/exploration_status`. On each `paused_for_detection` frame, captures RGB+depth, runs Detic (all classes), computes CLIP embeddings for each crop, projects to 3D world coordinates, and adds to the VoxelGraphBuilder.

2. **On exploration complete**: Node 1 calls `/scene_graph/build_and_rank`. The node:
   - Clusters raw detections with DBSCAN (eps=0.2m)
   - Builds a HeteroData scene graph with container, landmark, and room nodes
   - Computes soft-ldist spatial features
   - Runs the ContextGNN forward pass
   - Scores containers against the query embedding (cosine similarity)
   - Calls `/detection/set_rankings` on Node 2 with the sorted ranking list

## Imports from `semantic-object-container-room`

| Module | What's used |
|--------|------------|
| `realrobot.voxel_graph_builder` | `VoxelGraphBuilder` — accumulates detections, clusters, builds graph |
| `realrobot.inference` | `load_locked_model`, `score_containers`, `compute_soft_ldist` |
| `realrobot.detector` | `VisualDetector` (Detic 21K), `nms_by_type` |
| `realrobot.stretch.projection` | `project_bbox_to_world_se3` — SE(3) camera→world transform |
| `gnn.config` | `CLIP_DIM`, `QUERIES` |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `room_type` | `"kitchen"` | Room context for GNN |
| `query` | `"fork"` | Target object for scoring |
| `checkpoint` | `""` | GNN checkpoint path (empty = default) |
| `device` | `"cuda"` | Torch device for inference |
| `edge_cutoff` | `0.9` | Max distance for scene graph edges |
| `det_min_score` | `0.7` | Min Detic confidence |
| `use_sim` | `false` | Simulation mode |
| `robot_ip` | `""` | For rosbridge image transport (real robot) |

## Services

| Service | Type | Description |
|---------|------|-------------|
| `/scene_graph/build_and_rank` | Trigger | Build graph, run GNN, push rankings to Node 2 |
| `/scene_graph/get_rankings` | Trigger | Return current rankings as JSON |
| `/scene_graph/score_now` | Trigger | Force re-scoring (alias for build_and_rank) |

## Lazy Model Loading

Models are loaded on first use to reduce startup time:
- **GNN model** (~2MB): loaded to CPU first, moved to GPU only during scoring
- **Detic**: loaded on first detection frame
- **CLIP ViT-B/32**: loaded alongside Detic (~400MB VRAM)

## Sim vs Real

| Aspect | Simulation | Real Robot |
|--------|-----------|------------|
| Image transport | DDS (raw topics) | rosbridge WebSocket (compressed) |
| Image rotation | None | 90 CW (matches Node 2) |
| Camera K | From camera_info topic | Rotated intrinsics |
| TF source | Local TF buffer | Relies on Node 2's TF republishing |
| Runs on | Same machine as sim | GPU workstation (automata-3) |

## GNN Architecture

The ContextGNN uses:
- **Container nodes** (1540 dims): CLIP visual (512) + text type CLIP (512) + scene frame CLIP (512) + height encoding (4)
- **Room node** (1536 dims): room label CLIP + landmark mean + container mean
- **APPNP propagation**: K=3 steps with teleport
- **Cosine scoring**: container embeddings vs query embedding

## Known Limitations

- **GNN-drawer mismatch**: The GNN detects "Cabinet" or "Dresser" at a centroid. Node 2 detects individual "Drawer"+"Handle" at handle positions. A single GNN node may correspond to multiple drawers. Spatial matching handles this (one-to-many), but distances may be large. Check logs for match diagnostics.
- **VRAM**: Node 4 loads Detic + CLIP (~2GB). Combined with Node 2's Detic on the same GPU, ensure sufficient VRAM.
