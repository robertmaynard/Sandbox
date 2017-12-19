
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(DeviceLinking LANGUAGES CXX CUDA)

add_library(use_1 STATIC use_cublas_1.cu)
set_target_properties(use_1
                      PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON
                      POSITION_INDEPENDENT_CODE ON)

add_library(use_2 SHARED use_cublas_2.cu)
set_target_properties(use_2
                      PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

add_executable(valid_binary main.cu)
set_target_properties(valid_binary
                      PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(valid_binary PRIVATE use_1 use_2)

if(APPLE)
  # Help the static cuda runtime find the driver (libcuda.dyllib) at runtime.
  set_property(TARGET valid_binary PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
