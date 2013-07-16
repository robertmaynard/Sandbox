cmake_minimum_required(VERSION 2.8)
project(cudaPtxas)

find_package(CUDA)
mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD)


cuda_add_executable(cudaPtxas main.cu)
