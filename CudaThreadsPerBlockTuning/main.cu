
// Copyright 2017 Kitware, Inc.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "cuda.h"
#include <array>
#include <vector>
#include <iostream>

#include <vtkm/exec/internal/TaskSingular.h>

#include <vtkm/worklet/AverageByKey.h>
#include <vtkm/worklet/CellDeepCopy.h>
#include <vtkm/worklet/Clip.h>
#include <vtkm/worklet/MarchingCubes.h>

template<typename Task>
static __global__ void TaskStrided1DLaunch(vtkm::exec::internal::TaskSingular<Task> task, std::int64_t size) {
  const std::int64_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const std::int64_t inc = blockDim.x * gridDim.x;
  for(std::int64_t i=start; i < size; i+=inc)
  {
    task(i);
  }
}

template <typename Task> static void BlockSizeGuesser(int& grids, int& blocks, float& occupancy) {
  int blockSize;   // The launch configurator returned block size
  int minGridSize; // The minimum grid size needed to achieve the
                   // maximum occupancy for a full device launch
  int gridSize;    // The actual grid size needed, based on number of SM's
  int device;      // device to run on
  int numSMs;      // number of SMs on the active device

  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);

  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                     TaskStrided1DLaunch<Task>, 0, 0);


  blockSize /= (numSMs*2);
  gridSize = 32*numSMs;

  // calculate theoretical occupancy
  int maxActiveBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, TaskStrided1DLaunch<Task>, blockSize, 0);

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);

  grids = gridSize;
  blocks = blockSize;
  occupancy = (maxActiveBlocks * blockSize / props.warpSize) /
              (float)(props.maxThreadsPerMultiProcessor / props.warpSize);
}

template<typename Task> void compute(std::string name)
{
  int grids, blocks;
  float occupancy;

  BlockSizeGuesser<Task>(grids, blocks, occupancy);
  std::cout << name << ": " << " blocks of size " << blocks << " grid of size " << grids << std::endl;
  std::cout << name << ": " << " theoretical occupancy:" << occupancy << std::endl;
}


int main()
{
  compute<vtkm::worklet::DivideWorklet>("AverageByKey");
  compute<vtkm::worklet::CellDeepCopy::PassCellStructure>("CellDeepCopy");
  // compute<vtkm::worklet::Clip::ComputeStats<vtkm::cont::DeviceAdapterTagCuda>>("Clip ComputeStats");
  // compute<vtkm::worklet::Clip::GenerateCellSet<vtkm::cont::DeviceAdapterTagCuda>>("Clip GenerateCellSet");
  // compute<vtkm::worklet::Clip::marchingcubes::ClassifyCell<float>>("MC ClassifyCell");
  // compute<vtkm::worklet::Clip::marchingcubes::EdgeWeightGenerate<float,vtkm::cont::DeviceAdapterTagCuda>>("MC EdgeWeightGenerate");

}
