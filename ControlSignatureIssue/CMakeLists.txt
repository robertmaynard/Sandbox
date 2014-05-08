cmake_minimum_required(VERSION 2.8)

project(Dax)

find_package(Boost)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(Dax main.cxx)