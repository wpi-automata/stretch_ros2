# Evaluation

Full pipeline evaluation runs all three nodes together via a single launch file with testing flags set to false.

## Launch

### Simulation
```bash
ros2 launch stretch_drawer_pipeline pipeline_sim.launch.py
```

### Real Robot
```bash
ros2 launch stretch_drawer_pipeline pipeline_real.launch.py
```

## Evaluation Procedure

1. **Start exploration**:
   ```bash
   ros2 service call /mapping/start std_srvs/srv/Trigger
   ```

2. **Wait for mapping to complete**:
   ```bash
   # Poll until complete
   watch -n 2 "ros2 service call /mapping/is_complete std_srvs/srv/Trigger"
   ```

3. **Check detected drawers**:
   ```bash
   ros2 service call /detection/get_drawers std_srvs/srv/Trigger
   ```

4. **Execute drawer opening**:
   ```bash
   ros2 service call /navigate_open/execute std_srvs/srv/Trigger
   ```

5. **Monitor progress**:
   ```bash
   ros2 topic echo /navigate_open/status
   ```

## Metrics

### Mapping
- **Coverage**: Percentage of room area mapped (compare to known floor plan)
- **Time**: Seconds from start to exploration complete
- **Accuracy**: Alignment of occupancy grid to ground truth

### Detection
- **Precision**: Correctly identified drawers / total detections
- **Recall**: Correctly identified drawers / total actual drawers
- **Localization error**: Distance (m) between detected handle position and ground truth
- **De-duplication**: No duplicate markers for same physical drawer

### Navigation and Opening
- **Navigation success**: Robot reaches approach pose without collision
- **Grasp success**: Gripper makes contact and holds the handle
- **Open success**: Drawer physically opens
- **End-to-end time**: Seconds from execute call to drawer open

## Evaluation Environments

### Simulation (robocasa)
Test across multiple kitchen layouts and styles:
```bash
ros2 launch stretch_simulation stretch_mujoco_driver.launch.py \
    robocasa_task:=PnPCounterToCab \
    robocasa_layout:=<layout_id> \
    robocasa_style:=<style_id>
```

### Real Robot
Test in multiple physical rooms with different:
- Cabinet/drawer types (kitchen, bathroom, bedroom)
- Handle orientations (horizontal bars, vertical pulls, knobs)
- Heights (floor-level, counter-level, upper cabinets)

## Recording Results

Save a rosbag for each evaluation run:
```bash
ros2 bag record -o eval_run_001 \
    /map /drawer_markers /navigate_open/path \
    /navigate_open/status /exploration_status \
    /joint_states /tf /tf_static \
    /camera/color/image_raw
```

## Expected Baselines

| Metric | Target |
|--------|--------|
| Room coverage | > 90% |
| Drawer detection recall | > 80% |
| Drawer localization error | < 0.1m |
| Navigation success rate | > 90% |
| Grasp success rate | > 70% |
| End-to-end success | > 50% |
