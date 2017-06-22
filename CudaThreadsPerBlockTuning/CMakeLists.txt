cmake_minimum_required(VERSION 3.8)

project(CudaThreadsPerBlock CUDA CXX)

find_package(VTKm REQUIRED COMPONENTS CUDA)

add_executable(cuda_block_tuning main.cu)

target_compile_features(cuda_block_tuning PRIVATE cxx_nullptr)
target_compile_options(cuda_block_tuning PRIVATE -arch=sm_30 --resource-usage)
target_include_directories(cuda_block_tuning PRIVATE ${VTKm_INCLUDE_DIRS})

if(APPLE)
  # We need to add the default path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  target_link_libraries(cuda_block_tuning
                        PRIVATE
                        "-Wl,-rpath,${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}"
                        )
endif()
