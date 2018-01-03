cmake_minimum_required (VERSION 2.6)

project(CrossCompile)


message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_BASE_NAME: ${CMAKE_BASE_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "_CMAKE_TOOLCHAIN_PREFIX: ${_CMAKE_TOOLCHAIN_PREFIX}")
message(STATUS "CMAKE_INCLUDE_FLAG_CXX: ${CMAKE_INCLUDE_FLAG_CXX}")

message(STATUS "${CMAKE_SYSTEM_NAME}-${CMAKE_CXX_COMPILER_ID}-CXX")
add_executable(CrossCompile main.cpp)
target_include_directories(CrossCompile PUBLIC "/a/path/to/show")


