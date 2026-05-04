#!/usr/bin/env python3
"""
GNN Choose Drawers: explore a room with funmap, detect objects with DETIC 21K,
project to 3D, cluster with DBSCAN, build a heterogeneous scene graph, and
rank containers using a trained ContextGNN.

This node is perception + ranking ONLY.  It does NOT open, grasp, or
manipulate any drawer.  Its sole output is a ranked list of containers
(published as a JSON service and Rerun visualisation).

Pipeline (per camera frame):
  1. DETIC 21K detects all objects, classifies via ontology
  2. Detections projected to 3D via depth + SE(3) camera pose from TF2
  3. CLIP ViT-B/32 embeds each detection crop + scene frame
  4. Raw detections accumulated in VoxelGraphBuilder
  5. On scoring tick: DBSCAN clusters per type → instances → HeteroData
  6. ContextGNN scores containers for a query object
  7. Results logged to Rerun and published as ROS markers

Exploration: funmap frontier (head-scan → drive-to-scan loop).

Usage (simulation):
  # Terminal 1 — MuJoCo sim driver (use_cameras for point clouds, use_slam
  #   so the driver defers map→odom TF to funmap)
  ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_cameras:=true use_slam:=true

  # Terminal 2 — Funmap (fresh exploration, no pre-built map)
  ros2 launch stretch_demos gnn_funmap.launch.py

  # Terminal 3 — GNN Choose Drawers
  ros2 launch stretch_demos gnn_choose_drawers.launch.py room_type:=kitchen query:=fork

  # Optional: RViz
  ros2 run rviz2 rviz2

Usage (real robot):
  # Terminal 1 — Robot driver (d435i publishes point clouds automatically)
  ros2 launch stretch_core stretch_driver.launch.py

  # Terminal 2 — Funmap
  ros2 launch stretch_demos gnn_funmap.launch.py

  # Terminal 3 — GNN Choose Drawers
  ros2 launch stretch_demos gnn_choose_drawers.launch.py room_type:=kitchen query:=fork gnn_checkpoint:=/path/to/model.pt

Query the ranking from another terminal:
  ros2 service call /gnn_choose_drawers/get_ranking std_srvs/srv/Trigger

Subscribes:
  /camera/color/image_raw, /camera/depth/image_rect_raw, /camera/color/camera_info
  /gripper_camera/image_raw, /gripper_camera/depth/image_rect_raw, /gripper_camera/camera_info
  /stretch/joint_states

Calls:
  /funmap/trigger_head_scan  (std_srvs/Trigger)
  /funmap/trigger_drive_to_scan  (std_srvs/Trigger)

Provides:
  /gnn_choose_drawers/get_ranking  (std_srvs/Trigger → JSON)

Publishes:
  /gnn_choose_drawers/markers  (visualization_msgs/MarkerArray)
  /gnn_choose_drawers/detection_image  (sensor_msgs/Image)
"""

import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import rclpy.logging
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, JointState
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Trigger

import hello_helpers.hello_misc as hm

import torch

# -- Submodule imports (semantic-object-container-room) --
_SOCR_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'semantic-object-container-room'))
if _SOCR_ROOT not in sys.path:
    sys.path.insert(0, _SOCR_ROOT)

from realrobot.detector import VisualDetector, nms_by_type, draw_detections
from realrobot.stretch.projection import project_bbox_to_world_se3
from realrobot.voxel_graph_builder import VoxelGraphBuilder
from gnn.config import CLIP_DIM, EDGE_CUTOFF, QUERIES
from gnn.model import ContextGNN

# -- Camera topics --
HEAD_COLOR_TOPIC = '/camera/color/image_raw'
HEAD_DEPTH_TOPIC = '/camera/depth/image_rect_raw'
HEAD_INFO_TOPIC = '/camera/color/camera_info'
HEAD_OPTICAL_FRAME = 'camera_color_optical_frame'

WRIST_COLOR_TOPIC = '/gripper_camera/image_raw'
WRIST_DEPTH_TOPIC = '/gripper_camera/depth/image_rect_raw'
WRIST_INFO_TOPIC = '/gripper_camera/camera_info'
WRIST_OPTICAL_FRAME = 'gripper_camera_color_optical_frame'

# Head pan sweep for scanning
HEAD_PAN_POSITIONS = [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8]
HEAD_TILT_SEARCH = -0.5

MAX_FRONTIER_FAILURES = 3
MAX_PROJECTION_DEPTH = 6.0
SCORE_EVERY_N_STEPS = 3


def score_containers(model, graph, query_embeddings, query):
    if graph["container"].x.shape[0] == 0:
        return []
    model.eval()
    with torch.no_grad():
        out = model(graph, query_embeddings)
    scores = out["container_scores"][query].detach().cpu().numpy()
    types = graph.container_types
    order = np.argsort(scores)[::-1]
    return [
        {"rank": i + 1, "container_type": types[idx],
         "score": float(scores[idx]), "idx": int(idx)}
        for i, idx in enumerate(order)
    ]


class GNNChooseDrawersNode(hm.HelloNode):

    def __init__(self):
        hm.HelloNode.__init__(self)
        self.rate = 10.0
        self.bridge = CvBridge()
        self.callback_group = None

        # Camera data
        self.head_color = None
        self.head_depth = None
        self.head_info = None
        self.wrist_color = None
        self.wrist_depth = None
        self.wrist_info = None
        self.image_lock = threading.Lock()

        self.joint_states = None
        self.joint_states_lock = threading.Lock()

        # Perception models (loaded in main)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = None
        self.clip_model = None
        self.clip_preprocess = None
        self.builder = None
        self.gnn_model = None
        self.query_embeddings = None
        self.text_embeddings = None

        # Config
        self.room_type = 'kitchen'
        self.query = 'fork'
        self.det_min_score = 0.5
        self.gnn_checkpoint = ''
        self.rerun_enabled = True

        # Funmap clients
        self.head_scan_client = None
        self.drive_to_scan_client = None

        # State
        self.exploration_complete = False
        self.latest_ranking = []
        self.ranking_lock = threading.Lock()
        self.frame_counter = 0

        # Rerun
        self.rr = None

    # ── Callbacks ───────────────────────────────────────────────────────

    def joint_states_callback(self, msg):
        with self.joint_states_lock:
            self.joint_states = msg

    def head_color_cb(self, msg):
        with self.image_lock:
            self.head_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def head_depth_cb(self, msg):
        with self.image_lock:
            self.head_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def head_info_cb(self, msg):
        self.head_info = msg

    def wrist_color_cb(self, msg):
        with self.image_lock:
            self.wrist_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def wrist_depth_cb(self, msg):
        with self.image_lock:
            self.wrist_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def wrist_info_cb(self, msg):
        self.wrist_info = msg

    # ── Model loading ──────────────────────────────────────────────────

    def _load_models(self):
        self.logger.info('Loading DETIC 21K detector...')
        detic_root = os.path.expanduser('~/Detic')
        self.detector = VisualDetector(
            score_threshold=self.det_min_score,
            detic_root=detic_root,
        )

        self.logger.info('Loading CLIP ViT-B/32...')
        import clip as clip_module
        self.clip_model, self.clip_preprocess = clip_module.load(
            'ViT-B/32', device=self.device)

        self.logger.info('Building query embeddings...')
        query_tokens = clip_module.tokenize(
            [f'a photo of a {q}' for q in QUERIES]).to(self.device)
        with torch.no_grad():
            qe = self.clip_model.encode_text(query_tokens)
            qe = qe / qe.norm(dim=-1, keepdim=True)
        self.query_embeddings = {
            q: qe[i].float() for i, q in enumerate(QUERIES)}

        room_label = self.room_type.replace('_', ' ')
        tokens = clip_module.tokenize([f'a {room_label}']).to(self.device)
        with torch.no_grad():
            feat = self.clip_model.encode_text(tokens).squeeze(0)
            feat = feat / feat.norm()
        self.text_embeddings = {
            f'room_type:{self.room_type}': feat.cpu()}

        if self.gnn_checkpoint and os.path.exists(self.gnn_checkpoint):
            self.logger.info(f'Loading GNN from {self.gnn_checkpoint}')
            ckpt = torch.load(
                self.gnn_checkpoint, weights_only=False,
                map_location=self.device)
            self.gnn_model = ContextGNN(
                K=ckpt['K'], alpha=ckpt['alpha'], dropout=0.0)
            self.gnn_model.load_state_dict(ckpt['model_state_dict'])
            self.gnn_model.eval().to(self.device)
            self.logger.info(
                f'GNN loaded: K={ckpt["K"]}, alpha={ckpt["alpha"]}')
        else:
            self.logger.warn(
                'No GNN checkpoint — graph building only, no scoring')

        self.builder = VoxelGraphBuilder(
            scene_id=f'stretch_{self.room_type}',
            room_type=self.room_type,
            text_embeddings=self.text_embeddings,
            device=self.device,
            edge_cutoff=EDGE_CUTOFF,
            voxel_map=None,
        )
        self.logger.info('All models loaded')

    # ── Rerun ──────────────────────────────────────────────────────────

    def _init_rerun(self):
        import rerun as rr
        rr.init('gnn_choose_drawers', spawn=True)
        self.rr = rr
        self.logger.info('Rerun initialised')

    def _log_detection_rerun(self, det, world_pos, camera_name):
        if self.rr is None:
            return
        color_map = {
            'container': [255, 50, 50],
            'dual_role': [255, 165, 0],
            'landmark': [50, 100, 255],
            'object': [180, 180, 180],
        }
        color = color_map.get(det.category, [180, 180, 180])
        self.rr.log(
            f'world/{det.category}/{det.object_type}',
            self.rr.Points3D([world_pos.tolist()], colors=[color], radii=[0.04]),
        )

    def _log_frame_rerun(self, rgb_annotated, camera_name):
        if self.rr is None:
            return
        self.rr.log(
            f'cameras/{camera_name}/detections',
            self.rr.Image(np.array(rgb_annotated)),
        )

    def _log_ranking_rerun(self, ranking):
        if self.rr is None or not ranking:
            return
        lines = [f'{r["rank"]}. {r["container_type"]}: {r["score"]:+.3f}'
                 for r in ranking[:10]]
        self.rr.log(
            'scoring/ranking',
            self.rr.TextLog('\n'.join(lines)),
        )

    def _log_graph_rerun(self, graph):
        if self.rr is None:
            return
        if hasattr(graph['container'], 'pos') and graph['container'].pos is not None:
            pts = graph['container'].pos.numpy().tolist()
            self.rr.log(
                'world/graph/containers',
                self.rr.Points3D(pts, colors=[[255, 50, 50]] * len(pts), radii=[0.06]),
            )
        if hasattr(graph['landmark'], 'pos') and graph['landmark'].pos is not None:
            pts = graph['landmark'].pos.numpy().tolist()
            self.rr.log(
                'world/graph/landmarks',
                self.rr.Points3D(pts, colors=[[50, 100, 255]] * len(pts), radii=[0.06]),
            )

    # ── Per-frame processing ───────────────────────────────────────────

    def _clip_embed_crop(self, crop):
        if self.clip_model is None or crop is None:
            return None
        img_tensor = self.clip_preprocess(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(img_tensor).squeeze(0)
            emb = emb / emb.norm()
        return emb.cpu()

    def _clip_embed_full_frame(self, rgb):
        if self.clip_model is None:
            return None
        from PIL import Image as PILImage
        pil = PILImage.fromarray(rgb)
        img_tensor = self.clip_preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(img_tensor).squeeze(0)
            emb = emb / emb.norm()
        return emb.cpu()

    def process_frame(self, color, depth, camera_info, optical_frame, camera_name):
        if color is None or depth is None or camera_info is None:
            return 0

        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        dets = self.detector.detect(rgb, return_crops=True)
        dets = [d for d in dets if d.score >= self.det_min_score]
        dets = nms_by_type(dets)

        if not dets:
            return 0

        cam_to_odom, _ = hm.get_p1_to_p2_matrix(
            optical_frame, 'odom', self.tf2_buffer, timeout_s=1.0)
        if cam_to_odom is None:
            self.logger.info(f'[{camera_name}] TF lookup failed')
            return 0

        camera_K = np.array(camera_info.k).reshape(3, 3)

        n_added = 0
        scene_frame_emb = None

        for det in dets:
            world_pos = project_bbox_to_world_se3(
                det.bbox, depth, cam_to_odom, camera_K,
                max_depth=MAX_PROJECTION_DEPTH)
            if world_pos is None:
                continue

            clip_emb = self._clip_embed_crop(det.crop)

            x0, y0, x1, y1 = det.bbox
            area = (x1 - x0) * (y1 - y0)

            if det.category in ('container', 'dual_role'):
                if scene_frame_emb is None:
                    scene_frame_emb = self._clip_embed_full_frame(rgb)
                self.builder.add_container_observation(
                    obj_type=det.object_type,
                    position_3d=world_pos,
                    clip_embedding=clip_emb if clip_emb is not None else torch.zeros(CLIP_DIM),
                    scene_frame_clip=scene_frame_emb,
                    crop_area=area,
                )
                n_added += 1
            elif det.category == 'landmark':
                self.builder.add_landmark_observation(
                    obj_type=det.object_type,
                    position_3d=world_pos,
                    clip_embedding=clip_emb if clip_emb is not None else torch.zeros(CLIP_DIM),
                    crop_area=area,
                )
                n_added += 1

            self._log_detection_rerun(det, world_pos, camera_name)

        if dets:
            annotated = draw_detections(rgb, dets)
            self._log_frame_rerun(annotated, camera_name)
            try:
                det_msg = self.bridge.cv2_to_imgmsg(
                    np.array(annotated)[:, :, ::-1], encoding='bgr8')
                self.detection_pub.publish(det_msg)
            except Exception:
                pass

        self.frame_counter += 1
        return n_added

    # ── Camera scanning ────────────────────────────────────────────────

    def scan_with_cameras(self):
        self.move_to_pose({'joint_head_tilt': HEAD_TILT_SEARCH})
        time.sleep(0.3)

        for pan_angle in HEAD_PAN_POSITIONS:
            self.move_to_pose({'joint_head_pan': pan_angle})
            time.sleep(0.8)

            with self.image_lock:
                h_color = self.head_color.copy() if self.head_color is not None else None
                h_depth = self.head_depth.copy() if self.head_depth is not None else None

            n = self.process_frame(
                h_color, h_depth, self.head_info,
                HEAD_OPTICAL_FRAME, 'head')
            if n > 0:
                self.logger.info(f'[head pan={pan_angle:.1f}] {n} detections added')

        self.move_to_pose({'joint_head_pan': 0.0, 'joint_wrist_yaw': 1.57})
        time.sleep(0.8)

        with self.image_lock:
            w_color = self.wrist_color.copy() if self.wrist_color is not None else None
            w_depth = self.wrist_depth.copy() if self.wrist_depth is not None else None

        n = self.process_frame(
            w_color, w_depth, self.wrist_info,
            WRIST_OPTICAL_FRAME, 'wrist')
        if n > 0:
            self.logger.info(f'[wrist] {n} detections added')

    # ── Funmap frontier calls ──────────────────────────────────────────

    def _wait_for_future(self, future, timeout_sec):
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > timeout_sec:
                self.logger.warn(f'Service call timed out after {timeout_sec}s')
                return None
            time.sleep(0.1)
        return future.result()

    def call_funmap_head_scan(self):
        if not self.head_scan_client.wait_for_service(timeout_sec=5.0):
            self.logger.warn('/funmap/trigger_head_scan not available')
            return False
        future = self.head_scan_client.call_async(Trigger.Request())
        result = self._wait_for_future(future, timeout_sec=60.0)
        if result is not None:
            return result.success
        return False

    def call_funmap_drive_to_scan(self):
        if not self.drive_to_scan_client.wait_for_service(timeout_sec=5.0):
            self.logger.warn('/funmap/trigger_drive_to_scan not available')
            return False
        future = self.drive_to_scan_client.call_async(Trigger.Request())
        result = self._wait_for_future(future, timeout_sec=120.0)
        if result is not None:
            return result.success
        return False

    # ── GNN scoring ────────────────────────────────────────────────────

    def run_scoring(self):
        if self.gnn_model is None:
            return
        if self.builder.n_containers == 0:
            self.logger.info('No containers yet — skipping scoring')
            return

        graph = self.builder.get_graph().to(self.device)
        ranking = score_containers(
            self.gnn_model, graph, self.query_embeddings, self.query)

        with self.ranking_lock:
            self.latest_ranking = ranking

        nc = self.builder.n_containers
        nl = self.builder.n_landmarks
        nr = self.builder.n_raw_detections
        self.logger.info(
            f'Graph: {nc}C + {nl}L from {nr} raw detections')

        if ranking:
            top = ranking[0]
            self.logger.info(
                f'Top for "{self.query}": {top["container_type"]} '
                f'({top["score"]:+.3f})')
            for r in ranking[:5]:
                self.logger.info(
                    f'  #{r["rank"]} {r["container_type"]}: {r["score"]:+.3f}')

        self.publish_graph_markers(graph)
        self._log_ranking_rerun(ranking)
        self._log_graph_rerun(graph)

    # ── Markers ────────────────────────────────────────────────────────

    def publish_graph_markers(self, graph):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        if hasattr(graph['container'], 'pos') and graph['container'].pos is not None:
            for i in range(graph['container'].pos.shape[0]):
                pos = graph['container'].pos[i]
                ctype = graph.container_types[i] if i < len(graph.container_types) else '?'
                m = Marker()
                m.header.frame_id = 'odom'
                m.header.stamp = stamp
                m.ns = 'gnn_containers'
                m.id = i
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = float(pos[0])
                m.pose.position.y = float(pos[1])
                m.pose.position.z = float(pos[2])
                m.pose.orientation.w = 1.0
                m.scale.x = 0.15
                m.scale.y = 0.15
                m.scale.z = 0.15
                m.color.r = 1.0
                m.color.g = 0.3
                m.color.b = 0.1
                m.color.a = 0.8
                m.lifetime.sec = 0
                m.text = ctype
                marker_array.markers.append(m)

        if hasattr(graph['landmark'], 'pos') and graph['landmark'].pos is not None:
            for i in range(graph['landmark'].pos.shape[0]):
                pos = graph['landmark'].pos[i]
                m = Marker()
                m.header.frame_id = 'odom'
                m.header.stamp = stamp
                m.ns = 'gnn_landmarks'
                m.id = i
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(pos[0])
                m.pose.position.y = float(pos[1])
                m.pose.position.z = float(pos[2])
                m.pose.orientation.w = 1.0
                m.scale.x = 0.1
                m.scale.y = 0.1
                m.scale.z = 0.1
                m.color.r = 0.2
                m.color.g = 0.4
                m.color.b = 1.0
                m.color.a = 0.8
                m.lifetime.sec = 0
                marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)

    # ── Service: get ranking ───────────────────────────────────────────

    def get_ranking_callback(self, request, response):
        with self.ranking_lock:
            ranking = list(self.latest_ranking)
        response.success = len(ranking) > 0
        response.message = json.dumps({
            'query': self.query,
            'room_type': self.room_type,
            'n_raw_detections': self.builder.n_raw_detections,
            'n_containers': self.builder.n_containers,
            'n_landmarks': self.builder.n_landmarks,
            'ranking': ranking,
        })
        return response

    # ── Exploration loop ───────────────────────────────────────────────

    def exploration_loop(self):
        self.logger.info('Waiting for camera images...')
        for _ in range(200):
            with self.image_lock:
                if self.head_color is not None:
                    break
            time.sleep(0.1)

        if self.head_color is None:
            self.logger.error('No camera images received. Aborting.')
            return

        self.move_to_pose({
            'wrist_extension': 0.01,
            'joint_lift': 0.5,
            'joint_wrist_yaw': 0.0,
            'gripper_aperture': -0.05,
        })
        time.sleep(0.5)

        self.logger.info('Starting frontier exploration with funmap')
        consecutive_failures = 0
        scan_count = 0

        while rclpy.ok() and not self.exploration_complete:
            scan_count += 1
            self.logger.info(f'=== Frontier step {scan_count} ===')

            self.call_funmap_head_scan()
            time.sleep(0.5)

            self.scan_with_cameras()

            self.move_to_pose({
                'wrist_extension': 0.01,
                'joint_head_pan': 0.0,
                'joint_head_tilt': 0.0,
            })
            time.sleep(0.3)

            if scan_count % SCORE_EVERY_N_STEPS == 0:
                self.run_scoring()

            success = self.call_funmap_drive_to_scan()
            if success:
                consecutive_failures = 0
                time.sleep(0.5)
            else:
                consecutive_failures += 1
                self.logger.info(
                    f'No frontier ({consecutive_failures}/{MAX_FRONTIER_FAILURES})')
                if consecutive_failures >= MAX_FRONTIER_FAILURES:
                    self.logger.info('No more frontiers — exploration done')
                    self.exploration_complete = True

        # Final scoring
        self.logger.info('Running final GNN scoring...')
        self.run_scoring()

        with self.ranking_lock:
            ranking = list(self.latest_ranking)

        self.logger.info(f'Exploration complete. {self.builder.n_raw_detections} raw detections, '
                         f'{self.builder.n_containers}C + {self.builder.n_landmarks}L instances')
        if ranking:
            self.logger.info(f'Final ranking for "{self.query}":')
            for r in ranking:
                self.logger.info(
                    f'  #{r["rank"]} {r["container_type"]}: {r["score"]:+.3f}')

    # ── Node setup ─────────────────────────────────────────────────────

    def main(self):
        hm.HelloNode.main(
            self, 'gnn_choose_drawers', 'gnn_choose_drawers',
            wait_for_first_pointcloud=False)

        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Parameters — only declare if not already set by launch overrides
        for pname, default in [('room_type', 'kitchen'), ('query', 'fork'),
                               ('det_min_score', 0.5), ('gnn_checkpoint', ''),
                               ('rerun_enabled', True)]:
            if not self.has_parameter(pname):
                self.declare_parameter(pname, default)

        self.room_type = self.get_parameter(
            'room_type').get_parameter_value().string_value
        self.query = self.get_parameter(
            'query').get_parameter_value().string_value
        self.det_min_score = self.get_parameter(
            'det_min_score').get_parameter_value().double_value
        self.gnn_checkpoint = self.get_parameter(
            'gnn_checkpoint').get_parameter_value().string_value
        self.rerun_enabled = self.get_parameter(
            'rerun_enabled').get_parameter_value().bool_value

        self.logger.info(f'Room: {self.room_type}, Query: "{self.query}", '
                         f'Device: {self.device}')

        # Load perception models
        self._load_models()

        # Rerun
        if self.rerun_enabled:
            try:
                self._init_rerun()
            except Exception as e:
                self.logger.warn(f'Rerun init failed: {e}')

        # Joint states
        self.create_subscription(
            JointState, '/stretch/joint_states',
            self.joint_states_callback, qos_profile=0,
            callback_group=self.callback_group)

        # Head camera
        self.create_subscription(
            Image, HEAD_COLOR_TOPIC, self.head_color_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)
        self.create_subscription(
            Image, HEAD_DEPTH_TOPIC, self.head_depth_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)
        self.create_subscription(
            CameraInfo, HEAD_INFO_TOPIC, self.head_info_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)

        # Wrist camera
        self.create_subscription(
            Image, WRIST_COLOR_TOPIC, self.wrist_color_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)
        self.create_subscription(
            Image, WRIST_DEPTH_TOPIC, self.wrist_depth_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)
        self.create_subscription(
            CameraInfo, WRIST_INFO_TOPIC, self.wrist_info_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)

        # Funmap frontier services
        self.head_scan_client = self.create_client(
            Trigger, '/funmap/trigger_head_scan',
            callback_group=self.callback_group)
        self.drive_to_scan_client = self.create_client(
            Trigger, '/funmap/trigger_drive_to_scan',
            callback_group=self.callback_group)

        # Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray, '/gnn_choose_drawers/markers', 10,
            callback_group=self.callback_group)
        self.detection_pub = self.create_publisher(
            Image, '/gnn_choose_drawers/detection_image', 10,
            callback_group=self.callback_group)

        # Service
        self.create_service(
            Trigger, '/gnn_choose_drawers/get_ranking',
            self.get_ranking_callback,
            callback_group=self.callback_group)

        self.logger.info('GNNChooseDrawersNode ready. Starting exploration...')

        self.explore_thread = threading.Thread(
            target=self.exploration_loop, daemon=True)
        self.explore_thread.start()


def main():
    try:
        node = GNNChooseDrawersNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        rclpy.logging.get_logger('gnn_choose_drawers').info('Shutting down')


if __name__ == '__main__':
    main()
