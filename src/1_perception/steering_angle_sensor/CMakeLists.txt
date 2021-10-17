cmake_minimum_required(VERSION 2.8.3)
project(steering_angle_sensor)

set(PROJECT_DEPS
  roscpp
  std_msgs
  fsd_common_msgs
  message_runtime
  geometry_msgs
  )

find_package(catkin REQUIRED COMPONENTS
  message_generation  # Needed to generate custom message types
  std_msgs
  roscpp
  rospy
  geometry_msgs
  fsd_common_msgs
  message_runtime
  )

#add_message_files(  # Tell catkin which custom message files to include
#  FILES
#  ControlCommand.msg
#  )

#generate_messages(
#  DEPENDENCIES
#  std_msgs
#  geometry_msgs
#)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs # Needed to generate custom messages
)
include_directories(
  # include
  ${catkin_INCLUDE_DIRS}
  ${roscpp_INCLUDE_DIRS}
  ${rospy_INCLUDE_DIRS}
)

# Each node in the package must be declared like this
#add_executable(${PROJECT_NAME}
 # src/cone_detector_handle.cpp
  #src/cone_detector.cpp
  #src/main.cpp
  #)
#target_link_libraries(${PROJECT_NAME}
#  ${catkin_LIBRARIES}
#  )
