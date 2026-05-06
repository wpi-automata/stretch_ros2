# Node 3: Navigate and Open Drawer

## Overview

Given a list of detected drawers from Node 2, navigates the robot to the best target and executes a force-feedback grasp-and-pull sequence to open the drawer.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `approach_distance` | `0.45` | Meters from handle to position base |
| `grasp_force_threshold` | `5.0` | Force (N) to detect handle contact |
| `pull_force_threshold` | `15.0` | Force (N) to stop pulling |
| `gripper_close_effort` | `-50.0` | Effort for gripper close |
| `arm_extension_speed` | `0.01` | Meters per step approaching |
| `pull_speed` | `0.02` | Meters per step pulling |
| `max_pull_distance` | `0.4` | Max pull-back distance |
| `use_sim` | `false` | Simulation mode |

## Execution Sequence

1. **Select Drawer**: Get list from Node 2, sort by ranking (or distance if no ranking)
2. **Switch to position mode**: Call `/switch_to_position_mode` so `translate_mobile_base` and `rotate_mobile_base` commands move the base
3. **Navigate**: Rotate toward, then drive to `approach_distance` from handle using `rotate_mobile_base` and `translate_mobile_base` via FollowJointTrajectory
4. **Align**: Compute drawer face normal from corner geometry, rotate perpendicular so the arm faces the drawer
5. **Orient Wrist**: Rotate wrist yaw to match handle orientation (horizontal/vertical)
6. **Approach**: Incrementally extend arm until `grasp_force_threshold` exceeded
7. **Grasp**: Close gripper to hold handle
8. **Pull**: Retract arm until `pull_force_threshold` or `max_pull_distance`
9. **Release**: Open gripper and retract arm fully
10. **Switch to navigation mode**: Call `/switch_to_navigation_mode` in a `finally` block (always runs, even on failure/exception) so `cmd_vel` works again for frontier exploration

## Force Feedback

The node monitors `/joint_states` effort values:
- **Contact detection**: `wrist_extension` effort exceeds `grasp_force_threshold`
- **Pull limit**: `wrist_extension` effort exceeds `pull_force_threshold`

This works in both MuJoCo simulation (which provides simulated force readings) and on the real Stretch3 hardware.

## RViz Visualization

- Pink cube: Target drawer being opened, sized to match the drawer's world-frame bounding box
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
