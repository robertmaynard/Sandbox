cmake_minimum_required(VERSION 3.3)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
  set(CMAKE_CXX_EXTENSIONS FALSE)
endif()

project(AtC)

add_executable(AtC opt.cpp)
add_executable(AtC naive.cpp)
