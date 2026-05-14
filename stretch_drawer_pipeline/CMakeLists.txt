cmake_minimum_required(VERSION 3.8)
project(stretch_drawer_pipeline)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DrawerDetection.msg"
  "msg/DrawerList.msg"
  "srv/NavigateToDrawer.srv"
  "srv/TriggerDetection.srv"
  "srv/SetRankings.srv"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

# Install Python nodes
install(PROGRAMS
  stretch_drawer_pipeline/mapping_node.py
  stretch_drawer_pipeline/drawer_detection_node.py
  stretch_drawer_pipeline/navigate_open_node.py
  stretch_drawer_pipeline/exploration_node.py
  stretch_drawer_pipeline/scene_graph_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install rviz configs
install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

ament_package()
