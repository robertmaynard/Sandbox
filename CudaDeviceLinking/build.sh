#!/bin/bash
set -x

mkdir -p build

# build libuse_1 static
nvcc -arch sm_35 -Xcompiler=-fPIC -dc use_cublas_1.cu -o build/use_cublas_1.o
ar cr build/libuse_1.a build/use_cublas_1.o

# build libuse_2 static
nvcc -arch sm_35 -Xcompiler=-fPIC -dc use_cublas_2.cu -o build/use_cublas_2.o
ar cr build/libuse_2.a build/use_cublas_2.o

# build executable
nvcc -x cu -arch sm_35 -dc main.cu -o build/main.o
nvcc -v -dlink -arch sm_35 build/main.o -o build/cmake_device_link.o -L"$PWD/build" -luse_1 -luse_2 -lcublas_device -lcudadevrt -lcudart
c++ build/main.o build/cmake_device_link.o -o build/valid_binary -Wl,-rpath,/usr/local/cuda/lib -L"/usr/local/cuda/lib" -L"$PWD/build" -luse_1 -luse_2  -lcublas_device -lcudadevrt -lcudart
