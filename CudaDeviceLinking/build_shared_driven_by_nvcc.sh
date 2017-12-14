#!/bin/bash
set -x

mkdir -p build

# note this nvcc version only does a device link at the executable level

# build libuse_1 static
nvcc -arch sm_35 -Xcompiler=-fPIC -dc use_cublas_1.cu  -o build/use_cublas_1.o
nvcc -arch sm_35 -lib build/use_cublas_1.o -o build/libuse_1.a

# build libuse_2
nvcc -arch sm_35 -Xcompiler=-fPIC -dc use_cublas_2.cu  -o build/use_cublas_2.o
nvcc -v -arch sm_35 -shared build/use_cublas_2.o -o build/libuse_2.so -lcublas_device -lcudadevrt -lcudart

# build executable
nvcc -x cu -arch sm_35 -dc main.cu -o build/main.o
nvcc -v -arch sm_35 build/main.o -o  build/valid_binary -Xlinker=-rpath,/usr/local/cuda/lib -L"$PWD/build" -luse_1 -luse_2  -lcublas_device -lcudadevrt -lcudart
