cmake_minimum_required(VERSION 3.3)
project(EnableIfCompilerSlowdow)

set(CMAKE_CXX_STANDARD 11)
add_executable(fast fast.cpp)
add_executable(slow slow.cpp)
