cmake_minimum_required (VERSION 2.6)

project(CrossCompile C)

message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_BASE_NAME: ${CMAKE_BASE_NAME}")
message(STATUS "CMAKE_C_COMPILER_ID: ${CMAKE_C_COMPILER_ID}")
message(STATUS "_CMAKE_TOOLCHAIN_PREFIX: ${_CMAKE_TOOLCHAIN_PREFIX}")
message(STATUS "CMAKE_INCLUDE_FLAG_C: ${CMAKE_INCLUDE_FLAG_C}")

add_executable(CrossCompile main.c)
target_include_directories(CrossCompile PUBLIC "/a/path/to/show")


