cmake_minimum_required(VERSION 3.1)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
  set(CMAKE_CXX_EXTENSIONS FALSE)
endif()

project(t)

add_executable(old_style old.cpp)
target_include_directories(old_style PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(lazy lazy.cpp)
target_include_directories(lazy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(nonlazy nonlazy.cpp)
target_include_directories(nonlazy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

