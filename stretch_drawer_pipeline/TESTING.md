# Testing

Each node can be tested independently. This document covers both simulation and real robot testing.

## Prerequisites

### Simulation

All test launch files require the MuJoCo sim driver running first:

```bash
ros2 launch stretch_simulation stretch_mujoco_driver.launch.py \
  use_cameras:=true use_rviz:=false mode:=navigation
```

Always pass `use_sim:=true` when running against MuJoCo.

### Real Robot

Ensure the Stretch driver and rosbridge are running on the robot:
```bash
# On robot:
ros2 launch stretch_core stretch_driver.launch.py
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

For simple_explorer mode, also start the ZMQ server:
```bash
python -m stretch.app.zmq_server
```

### Cleanup

```bash
~/ament_ws/scripts/kill_ros.sh
```

---

## Node 1: Exploration

**Goal**: Verify the robot explores and pauses for detection at each scan point.

### Simulation (funmap mode)
```bash
ros2 launch stretch_drawer_pipeline pipeline_with_gnn.launch.py use_sim:=true
ros2 service call /mapping/start std_srvs/srv/Trigger
ros2 topic echo /exploration_status
```

### Real Robot (simple_explorer mode)
```bash
ros2 launch stretch_drawer_pipeline pipeline_with_gnn.launch.py \
  exploration_mode:=simple_explorer
ros2 service call /mapping/start std_srvs/srv/Trigger
```

### Success Criteria
- Status cycles through: `scanning` → `paused_for_detection` → `driving` → repeat
- Robot visits multiple positions
- Ends with `complete`
- Node 2 logs one detection pass per `paused_for_detection` event

---

## Node 2: Drawer Detection

**Goal**: Verify drawers are detected and markers appear in RViz.

### Launch (test mode — detects every frame, no exploration needed)
```bash
# Sim:
ros2 launch stretch_drawer_pipeline test_detection.launch.py use_sim:=true
# Real:
ros2 launch stretch_drawer_pipeline test_detection.launch.py
```

### Procedure
1. Open RViz, add `/drawer_markers` (MarkerArray)
2. Drive robot to face drawers
3. Check detections:
   ```bash
   ros2 service call /detection/get_drawers std_srvs/srv/Trigger
   ```

### Success Criteria
- Green markers = reachable, red = unreachable
- No duplicates for the same physical drawer
- Handle orientation labels correct

---

## Node 4: Scene Graph

**Goal**: Verify scene detections accumulate and GNN scoring works.

### Standalone test
```bash
# Launch scene graph node alone (needs camera topics):
ros2 run stretch_drawer_pipeline scene_graph_node.py \
  --ros-args -p use_sim:=true -p room_type:=kitchen -p query:=fork
```

### With exploration
```bash
ros2 launch stretch_drawer_pipeline pipeline_with_gnn.launch.py use_sim:=true
ros2 service call /mapping/start std_srvs/srv/Trigger
# Wait for exploration to complete, then check rankings:
ros2 service call /scene_graph/get_rankings std_srvs/srv/Trigger
```

### Manual scoring trigger
```bash
ros2 service call /scene_graph/score_now std_srvs/srv/Trigger
```

### Success Criteria
- Node logs detection counts per frame ("Frame N: X dets, Y added")
- `build_and_rank` returns success with top-ranked container
- `/detection/set_rankings` is called and Node 2 logs match results

---

## Node 3: Navigate and Open

**Goal**: Verify the robot navigates to a drawer and opens it.

### Launch
```bash
ros2 launch stretch_drawer_pipeline test_navigate.launch.py use_sim:=true
```

### Procedure
1. Drive robot near drawers, wait for drawer markers
2. Trigger:
   ```bash
   ros2 service call /navigate_open/execute std_srvs/srv/Trigger
   ```
3. Monitor:
   ```bash
   ros2 topic echo /navigate_open/status
   ```

### Success Criteria
- Robot navigates to drawer, aligns, extends arm, grasps, pulls
- Sim: fixed distance pull. Real: force-feedback pull

### Force feedback (real robot)
```bash
ros2 topic echo /joint_states --field effort
```

---

## Full Integration Test

### Simulation
```bash
ros2 launch stretch_simulation stretch_mujoco_driver.launch.py \
  use_cameras:=true use_rviz:=false mode:=navigation

ros2 launch stretch_drawer_pipeline pipeline_with_gnn.launch.py use_sim:=true
ros2 service call /mapping/start std_srvs/srv/Trigger

# Wait for exploration to complete, then:
ros2 service call /navigate_open/execute std_srvs/srv/Trigger
```

### Real Robot
```bash
# On robot:
ros2 launch stretch_core stretch_driver.launch.py
ros2 launch rosbridge_server rosbridge_websocket_launch.xml

# On automata-3:
ros2 launch stretch_drawer_pipeline pipeline_with_gnn.launch.py
ros2 service call /mapping/start std_srvs/srv/Trigger

# Wait for exploration, then:
ros2 service call /detection/choose_drawer std_srvs/srv/Trigger
ros2 service call /navigate_open/execute std_srvs/srv/Trigger
```

### What to Watch

1. **Exploration**: `/exploration_status` transitions (idle → scanning → paused → driving → complete)
2. **Detection**: Node 2 logs "Detection pass: N new" at each pause point
3. **Scene Graph**: Node 4 logs "Frame N: X dets, Y added" at each pause point
4. **GNN Scoring**: Node 4 logs "Top ranked: ContainerType (score=X.XXX)"
5. **Ranking Match**: Node 2 logs "Drawer ABC ← GNN XYZ (score=..., dist=...)"
6. **Navigation**: `/navigate_open/status` transitions through the state machine
