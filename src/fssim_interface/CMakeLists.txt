# AMZ-Driverless
# Copyright (c) 2018 Authors:
#   - Juraj Kabzan <kabzanj@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 2.8.3)

project(fssim_interface)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
#list(APPEND CMAKE_PREFIX_PATH "/home/harvey/turtlebot2/devel;")
#list(APPEND CMAKE_PREFIX_PATH "/opt/ros/melodic;")
#list(APPEND CMAKE_PREFIX_PATH "/home/harvey/fsd_skeleton/build/fssim_common")
#list(APPEND CMAKE_PREFIX_PATH "/home/harvey/fsd_fssim/build/fssim_common/catkin_generated")
#message( ${CMAKE_PREFIX_PATH})

#CMAKE_PREFIX_PATH = r'/home/harvey/turtlebot2/devel;/opt/ros/melodic;/home/harvey/fsd_fssim/src/fssim'.split(';')

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        nav_msgs
        tf
        fsd_common_msgs
        fssim_common)

catkin_package()

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${roscpp_INCLUDE_DIRS}
        ${rospy_INCLUDE_DIRS}
)

add_executable(fssim_interface_node src/fssim_interface_node.cpp)
target_link_libraries(fssim_interface_node ${catkin_LIBRARIES})
