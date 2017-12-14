
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void deviceCublasSgemm_1(int n, float alpha, float beta,
                                    const float *d_A, const float *d_B,
                                    float *d_C) {
  cublasHandle_t cnpHandle;
  cublasStatus_t status = cublasCreate(&cnpHandle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    return;
  }

  // Call function defined in the cublas_device system static library.
  // This way we can verify that we properly pass system libraries to the
  // device link line
  status = cublasSgemm(cnpHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                       d_A, n, d_B, n, &beta, d_C, n);

  cublasDestroy(cnpHandle);
}

void call_cublas_from_static_lib1() {
  // initial values that will make sure that the cublasSgemm won't actually
  // do any work
  int n = 0;
  float alpha = 1;
  float beta = 1;
  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;
  deviceCublasSgemm_1<<<1, 1>>>(n, alpha, beta, d_A, d_B, d_C);
}
