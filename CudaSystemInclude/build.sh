#!/bin/bash
set -x

mkdir -p build

# # build executable
nvcc -x cu -arch sm_30 -c main.cu -o build/main.o -isystem=/usr/local/cuda/include
nvcc -arch sm_30 build/main.o -o  build/system_include -Xlinker=-rpath,/usr/local/cuda/lib -isystem=/usr/local/cuda/include

