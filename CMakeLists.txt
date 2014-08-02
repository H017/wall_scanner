cmake_minimum_required(VERSION 2.8.3)
project(wall_scanner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  geometry_msgs
)

generate_messages(
  DEPENDENCIES
  nav_msgs
  geometry_msgs
)

catkin_package()