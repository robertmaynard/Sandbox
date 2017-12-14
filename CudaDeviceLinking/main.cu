
#include <cuda_runtime.h>
#include <iostream>

void call_cublas_from_static_lib1();
void call_cublas_from_static_lib2();

int choose_cuda_device()
{
  int nDevices = 0;
  cudaError_t err = cudaGetDeviceCount(&nDevices);
  if (err != cudaSuccess) {
    std::cerr << "Failed to retrieve the number of CUDA enabled devices"
              << std::endl;
    return 1;
  }
  for (int i = 0; i < nDevices; ++i) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, i);
    if (err != cudaSuccess) {
      std::cerr << "Could not retrieve properties from CUDA device " << i
                << std::endl;
      return 1;
    }

    if (prop.major > 3 || (prop.major == 3 && prop.minor >= 5)) {
      err = cudaSetDevice(i);
      if (err != cudaSuccess) {
        std::cout << "Could not select CUDA device " << i << std::endl;
      } else {
        return 0;
      }
    }
  }

  std::cout << "Could not find a CUDA enabled card supporting compute >=3.5"
            << std::endl;
  return 1;
}

int main(int argc, char** argv)
{
  int ret = choose_cuda_device();
  if (ret) {
    return 0;
  }

  call_cublas_from_static_lib1();
  call_cublas_from_static_lib2();

  return 0;
}
