# Testing

Each node can be tested independently. This document describes the test procedures.

## Node 1: Mapping and Exploration

**Goal**: Verify the robot can autonomously map a room until no frontiers remain.

### Launch
```bash
ros2 launch stretch_drawer_pipeline test_mapping.launch.py use_sim:=true
```

### Procedure
1. Start the mapping service:
   ```bash
   ros2 service call /mapping/start std_srvs/srv/Trigger
   ```
2. Open RViz and add:
   - `/map` (OccupancyGrid display)
   - `/voxel_markers` (MarkerArray)
   - TF display
3. Observe the robot rotating and navigating to frontiers
4. Monitor status:
   ```bash
   ros2 topic echo /exploration_status
   ```
5. The node reports `complete` when no more frontiers exist

### Success Criteria
- Occupancy grid fills in over time
- Robot visits multiple positions in the room
- Exploration terminates when the room is fully covered
- No collisions during exploration

### Testing frontier_method parameter
```bash
# Test with voxel-based frontiers
ros2 launch stretch_drawer_pipeline test_mapping.launch.py frontier_method:=voxel
```

---

## Node 2: Drawer Detection

**Goal**: Verify drawers are detected and displayed correctly in RViz.

### Launch (Test Mode)
```bash
ros2 launch stretch_drawer_pipeline test_detection.launch.py use_sim:=true
```

In test mode, the detection node processes every incoming frame (no need for the mapping node to be running). The user drives the robot manually.

### Procedure
1. Open RViz and add:
   - `/drawer_markers` (MarkerArray)
   - `/camera/color/image_raw` (Image display)
2. Drive the robot around using RViz "2D Goal Pose" or teleop:
   ```bash
   ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/stretch/cmd_vel
   ```
3. As drawers come into view, green/red markers appear in RViz
4. Verify detected drawers:
   ```bash
   ros2 service call /detection/get_drawers std_srvs/srv/Trigger
   ```

### Success Criteria
- Drawers visible in the camera produce markers in RViz
- Green markers = reachable, Red markers = unreachable (too high/low)
- No duplicate markers for the same physical drawer
- Handle orientation (H/V) labels are correct

---

## Node 3: Navigate and Open (Quick Test)

**Goal**: Verify the robot can navigate to a detected drawer and open it.

### Launch
```bash
ros2 launch stretch_drawer_pipeline test_navigate.launch.py use_sim:=true
```

This launches both the detection node (test mode) and the navigate node.

### Procedure
1. Drive the robot around the room using RViz goal poses
2. Wait until drawer markers appear (detection node is running)
3. Trigger navigation:
   ```bash
   ros2 service call /navigate_open/execute std_srvs/srv/Trigger
   ```
4. Monitor:
   ```bash
   ros2 topic echo /navigate_open/status
   ```
5. Observe in RViz:
   - Pink marker appears on the target drawer
   - Green path line shows planned route
   - Robot navigates, aligns, extends arm, grasps, pulls

### Success Criteria
- Robot navigates to within 1m of the drawer
- Robot rotates to face the drawer with its arm side
- Arm extends and makes contact with the handle
- Gripper closes on the handle
- Robot pulls back and drawer opens
- Robot releases and retracts arm

### Monitoring Force Feedback
```bash
ros2 topic echo /joint_states --field effort
```

---

## Full Integration Test

Run all three nodes together:
```bash
ros2 launch stretch_drawer_pipeline pipeline_sim.launch.py
ros2 service call /mapping/start std_srvs/srv/Trigger
```

Wait for mapping to complete, then:
```bash
ros2 service call /navigate_open/execute std_srvs/srv/Trigger
```

Or trigger manually during exploration for faster iteration.
