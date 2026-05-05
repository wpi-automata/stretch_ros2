# Node 3: Navigate and Open Drawer

## Overview

Given a list of detected drawers from Node 2, navigates the robot to the best target and executes a force-feedback grasp-and-pull sequence to open the drawer.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `approach_distance` | `1.0` | Meters from handle to position base |
| `grasp_force_threshold` | `5.0` | Force (N) to detect handle contact |
| `pull_force_threshold` | `15.0` | Force (N) to stop pulling |
| `gripper_close_effort` | `-50.0` | Effort for gripper close |
| `arm_extension_speed` | `0.01` | Meters per step approaching |
| `pull_speed` | `0.02` | Meters per step pulling |
| `max_pull_distance` | `0.4` | Max pull-back distance |
| `use_sim` | `false` | Simulation mode |

## Execution Sequence

1. **Select Drawer**: Get list from Node 2, sort by ranking (or distance if no ranking)
2. **Navigate**: Move base to `approach_distance` from handle, perpendicular to drawer face
3. **Align**: Fine-tune rotation so the arm faces the drawer
4. **Orient Wrist**: Rotate wrist yaw to match handle orientation (horizontal/vertical)
5. **Approach**: Incrementally extend arm until `grasp_force_threshold` exceeded
6. **Grasp**: Close gripper to hold handle
7. **Pull**: Retract arm until `pull_force_threshold` or `max_pull_distance`
8. **Release**: Open gripper and retract arm fully

## Force Feedback

The node monitors `/joint_states` effort values:
- **Contact detection**: `wrist_extension` effort exceeds `grasp_force_threshold`
- **Pull limit**: `wrist_extension` effort exceeds `pull_force_threshold`

This works in both MuJoCo simulation (which provides simulated force readings) and on the real Stretch3 hardware.

## RViz Visualization

- Pink cube: Target drawer being opened
- Green path: Planned robot path to approach pose
- Status text on `/navigate_open/status`

## Failure Modes

- No reachable drawers detected → FAILED
- Navigation timeout → FAILED
- No contact during arm extension → FAILED
- Stop service called → immediate halt

## Service Interface

```bash
# Execute full sequence
ros2 service call /navigate_open/execute std_srvs/srv/Trigger

# Emergency stop
ros2 service call /navigate_open/stop std_srvs/srv/Trigger
```
