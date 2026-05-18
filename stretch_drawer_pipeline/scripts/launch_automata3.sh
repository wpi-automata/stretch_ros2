#!/usr/bin/env bash
# Launch the drawer pipeline on automata-3 (GPU workstation).
#
# Usage:
#   ./launch_automata3.sh                    # default robot IP
#   ./launch_automata3.sh 192.168.1.100      # custom robot IP
#
# Prerequisites: robot-side launch_robot.sh already running.

set -euo pipefail

ROBOT_IP="${1:-130.215.12.63}"
ROOM="${2:-kitchen}"
QUERY="${3:-fork}"

SESSION="automata3_pipeline"

# Kill old session if it exists
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n detection

# --- Window 0: Drawer detection (Detic) ---
tmux send-keys -t "$SESSION:detection" \
  "ros2 run stretch_drawer_pipeline drawer_detection_node.py --ros-args -p test_mode:=true -p use_sim:=false -p robot_ip:=$ROBOT_IP -p keep_drawers_open:=false" Enter

sleep 2

# --- Window 1: Scene graph + GNN ---
tmux new-window -t "$SESSION" -n scene_graph
tmux send-keys -t "$SESSION:scene_graph" \
  "ros2 run stretch_drawer_pipeline scene_graph_node.py --ros-args -p room_type:=$ROOM -p query:=$QUERY -p robot_ip:=$ROBOT_IP -p use_sim:=false" Enter

# --- Window 2: RViz ---
tmux new-window -t "$SESSION" -n rviz
tmux send-keys -t "$SESSION:rviz" \
  "ros2 run rviz2 rviz2 -d \$(ros2 pkg prefix stretch_drawer_pipeline)/share/stretch_drawer_pipeline/rviz/mapping.rviz" Enter

# --- Window 3: Service calls (interactive) ---
tmux new-window -t "$SESSION" -n control
tmux send-keys -t "$SESSION:control" \
  "echo '--- Automata-3 Control ---'
echo 'Choose best drawer:'
echo '  ros2 service call /detection/choose_drawer std_srvs/srv/Trigger'
echo ''
echo 'Note: RViz here only shows scene graph markers + bboxes (DDS).'
echo 'For camera feeds, launch RViz on the stretch monitor.'" Enter

tmux select-window -t "$SESSION:control"
tmux attach -t "$SESSION"
