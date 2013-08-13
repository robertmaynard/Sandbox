cmake_minimum_required (VERSION 2.8)
project(TBBFindTest)

set(CMAKE_MODULE_PATH  ${TBBFindTest_SOURCE_DIR})


find_package(TBB QUIET REQUIRED)

include_directories(${TBB_INCLUDE_DIRS})

set(sources main.cxx)
add_executable(tbbTest ${sources})
target_link_libraries(tbbTest ${TBB_LIBRARIES})

message(STATUS "TBB_INCLUDE_DIRS: ${TBB_INCLUDE_DIRS}")
message(STATUS "TBB_LIBRARIES: ${TBB_LIBRARIES}")

message(STATUS "TBB_VERSION_MAJOR: ${TBB_VERSION_MAJOR}")
message(STATUS "TBB_VERSION_MINOR: ${TBB_VERSION_MINOR}")

message(STATUS "TBB_INTERFACE_VERSION: ${TBB_INTERFACE_VERSION}")
message(STATUS "TBB_COMPATIBLE_INTERFACE_VERSION: ${TBB_COMPATIBLE_INTERFACE_VERSION}")
