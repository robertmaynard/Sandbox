cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(ForceSystemInclude CXX CUDA)

add_library(base STATIC base.cu)
add_library(middle STATIC middle.cu)
add_library(end STATIC end.cu)

target_link_libraries(middle PUBLIC base)
target_link_libraries(end PUBLIC middle)

set_target_properties(base middle end PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(base PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_features(base PUBLIC cxx_std_11)


add_executable(run main.cxx)
target_link_libraries(run PUBLIC base middle end)
