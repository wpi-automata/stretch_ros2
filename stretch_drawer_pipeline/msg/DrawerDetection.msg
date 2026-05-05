# A single detected drawer with handle information

# Unique ID for this drawer (for de-duplication)
string drawer_id

# RGB image with drawer bbox and handle bbox superimposed
sensor_msgs/Image annotated_image

# Handle center in world frame
geometry_msgs/Point handle_center_world

# Handle grasp location in world frame
geometry_msgs/Point handle_grasp_world

# Whether the robot can physically reach this drawer
bool reachable

# Handle orientation: "horizontal" or "vertical"
string handle_orientation

# Ranking score (0.0 if TBD / not yet ranked)
float64 ranking

# Distance from robot base to handle (meters)
float64 distance_to_robot
