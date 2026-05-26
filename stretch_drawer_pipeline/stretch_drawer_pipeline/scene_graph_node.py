#!/usr/bin/env python3
"""Scene-graph ranker: Detic all-class detection, VoxelGraphBuilder, GNN scoring.

SceneGraphRanker is a plain Python class (not a ROS Node). It is owned by
DrawerDetectionNode, which passes in shared camera data, TF buffer, and the
Detic detector so nothing is duplicated.

The ranker accumulates Detic detections during exploration, builds a scene
graph via VoxelGraphBuilder, and runs the ContextGNN to rank containers by
how likely they are to hold the target object.
"""

import json
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

_SEMANTIC_ROOT = Path(__file__).resolve().parent.parent.parent / "semantic-object-container-room"
if str(_SEMANTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_ROOT))

MAX_PROJECTION_DEPTH = 5.0
MIN_PROJECTION_DEPTH = 0.1


def _bbox_iou(a, b):
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _cross_class_nms(detections, iou_threshold=0.3):
    """Suppress overlapping bboxes across all classes, keeping higher confidence."""
    if not detections:
        return detections
    sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
    keep = []
    for det in sorted_dets:
        suppressed = False
        for kept in keep:
            if _bbox_iou(det.bbox, kept.bbox) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(det)
    return keep


class SceneGraphRanker:
    """Builds a scene graph from Detic detections and runs GNN ranking.

    Not a ROS Node — receives dependencies from the owning node.
    """

    def __init__(self, *, logger, detector, device="cuda",
                 room_type="kitchen", query="fork", checkpoint="",
                 edge_cutoff=0.9, det_min_score=0.7,
                 rank_type="scene_graph"):
        self._logger = logger
        self._detector = detector
        self.device = device
        self.room_type = room_type
        self.query = query
        self.checkpoint = checkpoint
        self.edge_cutoff = edge_cutoff
        self.det_min_score = det_min_score
        self.rank_type = rank_type

        # Lazy-loaded models
        self._models_loaded = False
        self._clip_model = None
        self._clip_preprocess = None
        self._builder = None
        self._gnn_model = None
        self._proto_matrix = None
        self._gnn_cfg = None
        self._query_embeddings = None
        self._text_embeddings = None
        self._clip_dim = None
        self._device = device
        self._clip_text_query = None

        # Rankings
        self._rankings = []
        self._rankings_lock = threading.Lock()
        self._n_observations = 0

        # Detection tracking for visualization
        self._detected_objects = []
        self._detected_objects_lock = threading.Lock()

        self._logger.info(
            f"SceneGraphRanker created: room={room_type}, query={query}, "
            f"device={device}, rank_type={rank_type}"
        )

    # ── Model loading ───────────────────────────────────────────────

    def load_models(self):
        """Eagerly load CLIP + GNN models. Call during deferred init."""
        return self._ensure_models_loaded()

    def _ensure_models_loaded(self):
        if self._models_loaded:
            return True

        try:
            import torch
            import clip as clip_module

            device = self.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                self._logger.warn("CUDA not available — falling back to CPU")

            self._logger.info(f"Loading models (rank_type={self.rank_type})…")

            self._clip_model, self._clip_preprocess = clip_module.load(
                "ViT-B/32", device=device
            )

            from gnn.config import CLIP_DIM
            self._clip_dim = CLIP_DIM

            query_token = clip_module.tokenize(
                [f"a photo of a {self.query}"]
            ).to(device)
            with torch.no_grad():
                q_emb = self._clip_model.encode_text(query_token).squeeze(0)
                q_emb = q_emb / q_emb.norm()
            self._clip_text_query = q_emb.float().cpu()
            self._logger.info(f"CLIP text query computed for '{self.query}'")

            if self.rank_type == "scene_graph":
                from realrobot.voxel_graph_builder import VoxelGraphBuilder
                from realrobot.inference import load_locked_model
                from gnn.config import QUERIES

                checkpoint = self.checkpoint or None
                self._gnn_model, self._proto_matrix, self._gnn_cfg = load_locked_model(
                    checkpoint, "cpu"
                )
                ec = self._gnn_cfg.get("edge_cutoff", self.edge_cutoff)
                self._logger.info(
                    f"GNN loaded: K={self._gnn_cfg.get('K')}, "
                    f"alpha={self._gnn_cfg.get('alpha')}, ec={ec}"
                )

                query_tokens = clip_module.tokenize(
                    [f"a photo of a {q}" for q in QUERIES]
                ).to(device)
                with torch.no_grad():
                    qe = self._clip_model.encode_text(query_tokens)
                    qe = qe / qe.norm(dim=-1, keepdim=True)
                self._query_embeddings = {
                    q: qe[i].float() for i, q in enumerate(QUERIES)
                }

                room_tokens = clip_module.tokenize(
                    [f"a {self.room_type.replace('_', ' ')}"]
                ).to(device)
                with torch.no_grad():
                    feat = self._clip_model.encode_text(room_tokens).squeeze(0)
                    feat = feat / feat.norm()
                self._text_embeddings = {
                    f"room_type:{self.room_type}": feat.cpu()
                }

                self._builder = VoxelGraphBuilder(
                    scene_id=f"stretch_{self.room_type}",
                    room_type=self.room_type,
                    text_embeddings=self._text_embeddings,
                    device=device,
                    edge_cutoff=ec,
                )

            self._device = device
            self._models_loaded = True
            mode_label = "GNN + CLIP" if self.rank_type == "scene_graph" else "CLIP only"
            self._logger.info(f"{mode_label} models loaded")
            return True

        except Exception as e:
            self._logger.error(f"Failed to load models: {e}")
            return False

    # ── CLIP helper methods (used by all rank_type modes) ────────────

    def get_clip_text_query(self):
        """Return the pre-computed CLIP text embedding for the search query."""
        self._ensure_models_loaded()
        return self._clip_text_query

    def get_clip_text_embedding(self, text):
        """Encode a text string with CLIP and return L2-normalized CPU tensor."""
        import torch
        import clip as clip_module
        self._ensure_models_loaded()
        tokens = clip_module.tokenize([text]).to(self._device)
        with torch.no_grad():
            emb = self._clip_model.encode_text(tokens).squeeze(0)
            emb = emb / emb.norm()
        return emb.float().cpu()

    def get_clip_image_embedding(self, pil_image):
        """Encode a PIL image with CLIP and return L2-normalized CPU tensor."""
        import torch
        self._ensure_models_loaded()
        img_t = self._clip_preprocess(pil_image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._clip_model.encode_image(img_t).squeeze(0)
            emb = emb / emb.norm()
        return emb.float().cpu()

    # ── Cleanup (replaces /drawer_cleanup subscription) ─────────────

    def handle_cleanup(self, drawers):
        """Remove handle nodes that overlap with detected drawer handles.

        Args:
            drawers: list of dicts with keys handle_center (np.array),
                     drawer_center (np.array), dedup_distance (float).
        """
        if self.rank_type != "scene_graph":
            return
        if self._builder is None:
            return
        removed = 0
        for entry in drawers:
            hc = entry["handle_center"]
            dc = entry["drawer_center"]
            dedup = entry["dedup_distance"]

            has_drawer_match = False
            for node in self._builder.nodes.values():
                if node.node_type not in ("drawer", "Drawer"):
                    continue
                pos = np.array([node.position_3d["x"], node.position_3d["y"], node.position_3d["z"]])
                if np.linalg.norm(pos - dc) < dedup:
                    has_drawer_match = True
                    break

            if not has_drawer_match:
                continue

            to_remove = []
            for iid, node in self._builder.nodes.items():
                if node.node_type not in ("handle", "Handle"):
                    continue
                pos = np.array([node.position_3d["x"], node.position_3d["y"], node.position_3d["z"]])
                if np.linalg.norm(pos - hc) < dedup:
                    to_remove.append(iid)

            for iid in to_remove:
                self._builder.remove_node(iid)
                removed += 1

        if removed > 0:
            self._logger.info(
                f"Handle cleanup: removed {removed} handle node(s) from scene graph"
            )

    # ── Container injection from drawer detection node ───────────────

    def update_containers(self, containers):
        """Replace scene graph container nodes from the drawer detection node's list.

        Args:
            containers: list of dicts with keys:
                instance_id, obj_type, position_3d (np.ndarray),
                source_image (HxWx3 uint8 or None), drawer_bbox ([x0,y0,x1,y1]),
                score (float)
        """
        if not self._ensure_models_loaded():
            return

        import torch
        from PIL import Image

        built = []
        for c in containers:
            clip_emb = None
            scene_clip = None
            crop_area = 0.0

            bbox = c.get("drawer_bbox")
            src = c.get("source_image")
            if bbox is not None and src is not None:
                x0, y0, x1, y1 = bbox
                crop_area = (x1 - x0) * (y1 - y0)
                crop = src[y0:y1, x0:x1]
                if crop.size > 0:
                    pil_crop = Image.fromarray(crop)
                    img_t = self._clip_preprocess(pil_crop).unsqueeze(0).to(self._device)
                    with torch.no_grad():
                        clip_emb = self._clip_model.encode_image(img_t).squeeze(0)
                        clip_emb = (clip_emb / clip_emb.norm()).cpu()

                pil_frame = Image.fromarray(src)
                frame_t = self._clip_preprocess(pil_frame).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    scene_clip = self._clip_model.encode_image(frame_t).squeeze(0)
                    scene_clip = (scene_clip / scene_clip.norm()).cpu()

            built.append({
                "instance_id": c["instance_id"],
                "obj_type": c["obj_type"],
                "position_3d": c["position_3d"],
                "clip_embedding": clip_emb,
                "scene_frame_clip": scene_clip,
                "crop_area": crop_area,
                "score": c.get("score", 0.0),
            })

        self._ensure_type_text_embeddings([c["obj_type"] for c in containers])
        self._builder.set_containers(built)
        self._logger.info(
            f"Containers updated from drawer list: {len(built)} containers"
        )

    # ── Per-frame processing ────────────────────────────────────────

    def process_frame(self, rgb, depth, camera_pose, camera_K):
        """Process one camera frame: Detic → CLIP → 3D project → builder.

        Args:
            rgb: HxWx3 uint8 numpy array (already rotated)
            depth: HxW float32 numpy array in meters
            camera_pose: 4x4 SE(3) matrix, camera in odom frame
            camera_K: 3x3 intrinsics matrix (post-rotation)

        Returns:
            (n_nodes, n_observations) tuple.
        """
        if self.rank_type != "scene_graph":
            return (0, 0)

        import torch
        from PIL import Image

        if not self._ensure_models_loaded():
            return (0, 0)

        from realrobot.detector import nms_by_type
        from realrobot.stretch.projection import project_bbox_to_world_se3

        dets = self._detector.detect(rgb, return_crops=True)
        dets = [d for d in dets if d.score >= self.det_min_score]
        n_raw = len(dets)
        dets = nms_by_type(dets)
        n_nms1 = len(dets)
        dets = _cross_class_nms(dets, iou_threshold=0.3)
        self._logger.info(
            f"Detic: {n_raw} raw → {n_nms1} after per-type NMS → {len(dets)} after cross-class NMS")
        for d in dets:
            self._logger.info(
                f"  kept: {d.object_type} bbox={d.bbox} score={d.score:.2f}")

        if dets:
            self._ensure_type_text_embeddings(
                [d.object_type for d in dets]
            )

        n_added = 0
        scene_frame_emb = None

        for det in dets:
            world_pos = project_bbox_to_world_se3(
                det.bbox, depth, camera_pose, camera_K,
                max_depth=MAX_PROJECTION_DEPTH,
                image_rotated_cw90=True,
            )
            if world_pos is None:
                continue

            self._logger.info(
                f"  {det.object_type} @ ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})"
                f"  bbox={det.bbox}  score={det.score:.2f}"
            )

            clip_emb = None
            if det.crop is not None:
                img_t = self._clip_preprocess(det.crop).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    clip_emb = self._clip_model.encode_image(img_t).squeeze(0)
                    clip_emb = clip_emb / clip_emb.norm()
                clip_emb = clip_emb.cpu()

            if scene_frame_emb is None:
                ft = self._clip_preprocess(
                    Image.fromarray(rgb)
                ).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    scene_frame_emb = self._clip_model.encode_image(ft).squeeze(0)
                    scene_frame_emb = scene_frame_emb / scene_frame_emb.norm()
                scene_frame_emb = scene_frame_emb.cpu()

            x0, y0, x1, y1 = det.bbox
            area = (x1 - x0) * (y1 - y0)

            n_before = self._builder.n_nodes
            self._builder.add_observation(
                obj_type=det.object_type,
                position_3d=world_pos,
                clip_embedding=clip_emb if clip_emb is not None else torch.zeros(self._clip_dim),
                scene_frame_clip=scene_frame_emb,
                crop_area=area,
                score=det.score,
            )
            n_after = self._builder.n_nodes
            if n_after > n_before:
                self._logger.info(
                    f"    NEW node for {det.object_type} (total {n_after})")
            else:
                self._logger.info(
                    f"    MERGED {det.object_type} into existing node")
            n_added += 1

            with self._detected_objects_lock:
                self._detected_objects.append({
                    "type": det.object_type,
                    "pos": world_pos.tolist() if hasattr(world_pos, "tolist") else list(world_pos),
                    "score": det.score,
                })

        self._n_observations += 1
        self._logger.info(
            f"Frame {self._n_observations}: {len(dets)} dets, {n_added} added, "
            f"{self._builder.n_raw_detections} total raw, "
            f"{self._builder.n_nodes} nodes"
        )
        return (self._builder.n_nodes, self._n_observations)

    def _ensure_type_text_embeddings(self, obj_types):
        import torch
        import clip as clip_module
        from realrobot.voxel_graph_builder import CONTAINER_TYPES, _is_landmark, _is_dual_role

        needed = {}
        for t in obj_types:
            if t in CONTAINER_TYPES:
                needed.setdefault(f"container_type:{t}", t)
                if _is_dual_role(t):
                    needed.setdefault(f"landmark_type:{t}", t)
            elif _is_landmark(t):
                needed.setdefault(f"landmark_type:{t}", t)

        new_keys = [k for k in needed if k not in self._text_embeddings]
        if not new_keys:
            return
        prompts = [f"a photo of a {needed[k]}" for k in new_keys]
        tokens = clip_module.tokenize(prompts).to(self._device)
        with torch.no_grad():
            embs = self._clip_model.encode_text(tokens)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        for i, key in enumerate(new_keys):
            self._text_embeddings[key] = embs[i].float().cpu()

    # ── Query matching ─────────────────────────────────────────────

    def check_labels_against_query(self, threshold=0.8):
        """Check if any scene graph landmark label matches the query.

        Returns (label, score) for the first match above threshold, or None.
        """
        import torch
        if not self._ensure_models_loaded():
            return None
        if self._builder is None:
            return None
        if self._builder._dirty:
            self._builder._recluster()
        seen = set()
        for node in self._builder.landmark_nodes.values():
            if node.node_type in seen:
                continue
            seen.add(node.node_type)
            label_emb = self.get_clip_text_embedding(node.node_type)
            score = float(torch.dot(label_emb, self._clip_text_query))
            if score > threshold:
                return (node.node_type, score)
        return None

    def check_items_against_query(self, items, threshold=0.8):
        """Check if any item label matches the query.

        Args:
            items: list of dicts with 'label' key
            threshold: cosine similarity threshold

        Returns (label, score) for the first match above threshold, or None.
        """
        import torch
        if not self._ensure_models_loaded():
            return None
        for item in items:
            item_emb = self.get_clip_text_embedding(item["label"])
            score = float(torch.dot(item_emb, self._clip_text_query))
            if score > threshold:
                return (item["label"], score)
        return None

    # ── GNN scoring ──────────────────────────────────────────────────

    def build_and_rank(self) -> list:
        """Build scene graph and run GNN. Returns ranking list."""
        if self.rank_type != "scene_graph":
            return []

        import torch
        from realrobot.inference import score_containers

        if not self._ensure_models_loaded():
            return []

        if self._builder.n_nodes == 0:
            self._logger.warn("No nodes in scene graph — cannot score")
            return []

        self._logger.info(
            f"Running GNN scoring: {self._builder.n_nodes} nodes, "
            f"{self._builder.n_raw_detections} raw detections"
        )

        self._gnn_model.to(self._device)
        self._proto_matrix = self._proto_matrix.to(self._device)

        graph = self._builder.get_graph()
        ranking = score_containers(
            self._gnn_model, graph, self._query_embeddings,
            self.query, self._proto_matrix,
        )

        self._gnn_model.to("cpu")
        self._proto_matrix = self._proto_matrix.to("cpu")

        with self._rankings_lock:
            self._rankings = ranking

        if ranking:
            top = ranking[0]
            self._logger.info(
                f"Top ranked: {top.get('container_type')} "
                f"(score={top.get('score', 0):.3f})"
            )

        self._auto_save_state()
        return ranking

    def _auto_save_state(self):
        import torch
        from pathlib import Path
        save_dir = Path("/home/ros2_stretch/ament_ws/src/stretch_ros2/stretch_drawer_pipeline/stretch_drawer_pipeline/runs/scenegraphs")
        save_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(save_dir.glob("scene_graph_state_*.pt"))
        idx = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 0
        save_path = save_dir / f"scene_graph_state_{idx:03d}.pt"
        try:
            det_list = []
            for d in self._builder._detections:
                det_list.append({
                    "obj_type": d.obj_type,
                    "position": d.position.tolist() if hasattr(d.position, 'tolist') else list(d.position),
                    "clip_embedding": d.clip_embedding,
                    "scene_frame_clip": d.scene_frame_clip,
                    "crop_area": d.crop_area,
                    "score": d.score,
                })
            state = {
                "scene_id": self._builder.scene_id,
                "room_type": self._builder.room_type,
                "room_id": self._builder.room_id,
                "edge_cutoff": self._builder.edge_cutoff,
                "dbscan_eps": self._builder.dbscan_eps,
                "min_obs": self._builder.min_obs,
                "text_embeddings": self._builder.text_embeddings,
                "detections": det_list,
                "query": self.query,
                "query_embeddings": {q: e.cpu() for q, e in self._query_embeddings.items()},
                "proto_matrix": self._proto_matrix.cpu(),
                "gnn_cfg": self._gnn_cfg,
                "gnn_checkpoint": self.checkpoint,
                "rankings": self._rankings,
            }
            torch.save(state, save_path)
            self._logger.info(
                f"Scene graph state saved: {len(det_list)} detections, "
                f"{len(self._query_embeddings)} queries → {save_path}")
        except Exception as e:
            self._logger.warn(f"Failed to save scene graph state: {e}")

    def get_rankings(self) -> list:
        """Return current rankings."""
        with self._rankings_lock:
            return list(self._rankings)

    # ── Marker data (for the owning node to publish) ─────────────────

    def get_marker_data(self):
        """Return data needed by the owning node to publish RViz markers.

        Returns dict with keys: container_nodes, landmark_nodes, rankings,
        room_type, edge_cutoff.
        """
        if self.rank_type != "scene_graph":
            return {
                "container_nodes": [],
                "landmark_nodes": [],
                "rankings": [],
                "room_type": self.room_type,
                "edge_cutoff": 2.0,
            }
        if self._builder:
            if self._builder._dirty:
                self._builder._recluster()
            container_nodes = list(self._builder.container_nodes.values())
            landmark_nodes = list(self._builder.landmark_nodes.values())
            edge_cutoff = self._builder.edge_cutoff
        else:
            container_nodes = []
            landmark_nodes = []
            edge_cutoff = 2.0
        with self._rankings_lock:
            rankings = list(self._rankings)
        return {
            "container_nodes": container_nodes,
            "landmark_nodes": landmark_nodes,
            "rankings": rankings,
            "room_type": self.room_type,
            "edge_cutoff": edge_cutoff,
        }

    @property
    def n_nodes(self):
        return self._builder.n_nodes if self._builder else 0

    @property
    def n_observations(self):
        return self._n_observations
