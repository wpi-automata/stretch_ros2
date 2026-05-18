#!/usr/bin/env bash
# Launch the drawer pipeline on the Stretch robot.
#
# Usage:
#   ./launch_robot.sh              # default: simple mode, 1 position
#   ./launch_robot.sh funmap 4     # funmap mode, 4 positions
#
# Prerequisites: stretch_core, rosbridge, realsense packages installed.
# After all windows are up, start exploration with:
#   ros2 service call /mapping/start std_srvs/srv/Trigger

set -euo pipefail

MODE="${1:-simple}"
N_POS="${2:-1}"

SESSION="stretch_pipeline"

# Kill old session if it exists
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n drivers

# --- Window 0: stretch home + driver ---
tmux send-keys -t "$SESSION:drivers" \
  "stretch home && ros2 launch stretch_core stretch_driver.launch.py mode:=navigation broadcast_odom_tf:=True" Enter

sleep 2

# --- Window 1: TF republisher ---
tmux new-window -t "$SESSION" -n tf
tmux send-keys -t "$SESSION:tf" \
  "python3 ~/ament_ws/src/stretch_ros2/stretch_drawer_pipeline/scripts/tf_static_republisher.py" Enter

# --- Window 2: RealSense camera ---
tmux new-window -t "$SESSION" -n camera
tmux send-keys -t "$SESSION:camera" \
  "ros2 launch stretch_core stretch_realsense.launch.py align_depth.enable:=true pointcloud.enable:=true pointcloud.stream_filter:=2 pointcloud.ordered_pc:=true" Enter

# --- Window 3: Rosbridge ---
tmux new-window -t "$SESSION" -n rosbridge
tmux send-keys -t "$SESSION:rosbridge" \
  "ros2 launch rosbridge_server rosbridge_websocket_launch.xml" Enter

# --- Window 4: Pipeline nodes ---
tmux new-window -t "$SESSION" -n pipeline
tmux send-keys -t "$SESSION:pipeline" \
  "ros2 launch stretch_drawer_pipeline pipeline_robot.launch.py exploration_mode:=$MODE n_positions:=$N_POS" Enter

# --- Window 5: Service calls (interactive) ---
tmux new-window -t "$SESSION" -n control
tmux send-keys -t "$SESSION:control" \
  "echo '--- Pipeline Control ---'
echo 'Start exploration:'
echo '  ros2 service call /mapping/start std_srvs/srv/Trigger'
echo ''
echo 'Choose best drawer (after exploration completes):'
echo '  ros2 service call /detection/choose_drawer std_srvs/srv/Trigger'
echo ''
echo 'Open drawer (after navigate_open_node receives drawer):'
echo '  ros2 service call /navigate_open/execute std_srvs/srv/Trigger'
echo ''
echo 'Stop exploration early:'
echo '  ros2 service call /mapping/stop std_srvs/srv/Trigger'" Enter

tmux select-window -t "$SESSION:control"
tmux attach -t "$SESSION"
