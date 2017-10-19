#!/bin/bash

build="/home/kitware/buildslave/root/vtk-m-renar-linux-static-release_64bit_ids_benchmark_clang_cuda_host_gcc_6_cuda_native_examples_tbb/build/"

cur_dir="/home/kitware/benchmarking"
env_py="${cur_dir}/env/bin/python"
runner="${cur_dir}/benchmark_runner.py"
uploader="${cur_dir}/benchmark_uploader.py"
tmp_dir="${cur_dir}/logs/"
creds="${cur_dir}/credentials"

if [ -d "$build" ]; then
  cd $build
  echo "starting primary engines"
  ${env_py} ${runner} -d ${build} -p Benchmark -o ${tmp_dir} -r
  echo "firing secondary engines"
  ${env_py} ${uploader} ${tmp_dir} vtk-benchmark-outputs --credentials ${creds} --today
  echo "final velocity achieved"
fi
