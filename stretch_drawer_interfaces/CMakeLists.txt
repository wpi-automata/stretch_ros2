cmake_minimum_required(VERSION 3.8)
project(stretch_drawer_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/DrawerGrasp.srv"
  DEPENDENCIES sensor_msgs geometry_msgs
)

ament_package()
