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

## Launch Arguments

`approach_distance` and `max_pull_distance` are exposed as launch arguments for easy override:

```bash
ros2 launch stretch_drawer_pipeline test_navigate.launch.py use_sim:=true approach_distance:=0.6 max_pull_distance:=0.3
```

| Launch Argument | Default | Description |
|-----------------|---------|-------------|
| `use_sim` | `false` | Simulation mode (enables use_sim_time) |
| `approach_distance` | `0.45` | Distance (m) from handle to position robot base |
| `max_pull_distance` | `0.4` | Max distance (m) to retract arm when pulling drawer |

All other parameters can be tuned in `config/navigate_params.yaml`.

## Execution Sequence

1. **Select Drawer**: Get list from Node 2, sort by ranking (or distance if no ranking)
2. **Switch to position mode**: Call `/switch_to_position_mode` so `translate_mobile_base` and `rotate_mobile_base` commands move the base
3. **Navigate & Align**: Compute approach pose directly in front of the handle, offset along the drawer face normal by `approach_distance`. Execute rotate → drive → rotate to position the robot with its arm facing the handle
4. **Orient Wrist**: Set wrist yaw for bump approach (pi/2), roll based on handle orientation
5. **Approach (bump)**: Extend arm toward handle to find bump/contact point. Sim: computed distance. Real: force threshold
6. **Retract**: Pull arm back fully
7. **Open & Orient**: Open gripper, set wrist yaw to 0 (inline with arm) and roll for handle orientation
8. **Re-extend**: Extend arm back to the recorded bump point
9. **Grasp**: Close gripper to hold handle
10. **Pull**: Retract arm up to `max_pull_distance`. Sim: single retract command. Real: incremental with force threshold
11. **Release**: Open gripper and retract arm fully
12. **Look at drawer**: Pan/tilt head camera to view the opened drawer using TF geometry
13. **Switch to navigation mode**: Call `/switch_to_navigation_mode` in a `finally` block (always runs, even on failure/exception) so `cmd_vel` works again for frontier exploration

## Sim vs Real Differences

| Step | Simulation | Real Robot |
|------|-----------|------------|
| **Approach (bump)** | Computes distance to handle, extends directly | Extends incrementally, stops on force contact |
| **Pull** | Retracts full `max_pull_distance` in one command | Retracts incrementally, stops when effort > `pull_force_threshold` |
| **Force feedback** | MuJoCo publishes zero effort — unavailable | Monitors `/joint_states` effort for `grasp_force_threshold` and `pull_force_threshold` |
| **Arm extension joints** | Reads `joint_arm_l0..l3` (sum = total extension) | Reads `wrist_extension` directly |

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
