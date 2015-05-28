cmake_minimum_required(VERSION 2.8)

project(CudaStackCopy CXX)


find_package(VTKm REQUIRED)
include(VTKmMacros)
vtkm_configure_device(Cuda)

cuda_add_executable(CudaStackCopy main.cu)