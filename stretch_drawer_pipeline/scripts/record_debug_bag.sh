#!/bin/bash
# Record a 1-minute rosbag with all topics needed to debug the
# drawer detection and navigate_open pipeline.
#
# Run on the ROBOT (stretch-se3-3096) where all raw topics originate.
#
# Usage: bash record_debug_bag.sh [output_dir]

OUTPUT_DIR="${1:-/tmp/drawer_debug_$(date +%Y%m%d_%H%M%S)}"
DURATION=60

echo "Recording ${DURATION}s rosbag to: ${OUTPUT_DIR}"
echo "Press Ctrl+C to stop early."

ros2 bag record -o "$OUTPUT_DIR" --max-cache-size 500000000 -d "$DURATION" \
    /camera/color/image_raw \
    /camera/color/image_raw/compressed \
    /camera/color/camera_info \
    /camera/depth/image_rect_raw \
    /camera/aligned_depth_to_color/image_raw \
    /tf \
    /tf_static \
    /joint_states \
    /stretch/joint_states \
    /drawer_detections_json \
    /drawer_markers \
    /navigate_open/status \
    /navigate_open/path \
    /navigate_open/target_marker \
    /robot_description

echo "Done. Bag saved to: ${OUTPUT_DIR}"
