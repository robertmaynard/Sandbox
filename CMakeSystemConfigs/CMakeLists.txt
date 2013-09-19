cmake_minimum_required(VERSION 2.8)
project(Example)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

#get the machine name
site_name(computer_name)

#find the specific machine
include(${computer_name}.cmake OPTIONAL)