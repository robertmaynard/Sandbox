cmake_minimum_required(VERSION 3.1)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
  set(CMAKE_CXX_EXTENSIONS FALSE)
endif()

project(t)

add_library(default SHARED test.cpp)

add_library(unnamed SHARED test.cpp)
target_compile_definitions(unnamed PRIVATE USE_UNNAMED)

add_library(inline SHARED test.cpp)
target_compile_definitions(inline PRIVATE USE_INLINE)

add_library(static_inline SHARED test.cpp)
target_compile_definitions(static_inline PRIVATE USE_STATIC_INLINE)

add_library(force_inline SHARED test.cpp)
target_compile_definitions(force_inline PRIVATE USE_FORCE_INLINE)

