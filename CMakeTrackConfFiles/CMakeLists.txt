#tracking of configured files was only introduced in CMake 2.8.12
cmake_minimum_required(VERSION 2.8.12)

project(CMakeTrackConfFiles CXX)

#configure a file that we want to track
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/configured_file.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/configured_file.h
               COPYONLY)

#set the GENERATED flag on this other source file
set_property(SOURCE main.cxx PROPERTY GENERATED)


#configure and than delete a file that we don't want to track
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/to_delete.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/to_delete.h
               COPYONLY)
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/to_delete.h)


add_executable(Example main.cxx configured_file.h)
target_include_directories(Example PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

