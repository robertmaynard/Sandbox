cmake_minimum_required(VERSION 3.13...3.15 FATAL_ERROR)

project(bench CXX)

#Find the VTK-m package
find_package(VTKm REQUIRED QUIET)

add_executable(bench virtuals_cast_benchmark.cxx)
target_link_libraries(bench PUBLIC vtkm_cont)
target_include_directories(bench PUBLIC "/home/robert/Work/misc/benchmark/install/include/")
target_link_directories(bench PUBLIC "/home/robert/Work/misc/benchmark/install/lib")
target_link_libraries(bench PUBLIC  -lbenchmark -pthread)
