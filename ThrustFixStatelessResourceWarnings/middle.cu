
#include <cooperative_groups.h>
#include <cuda.h>

#include <thrust/advance.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cpp/memory.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/unique.h>

#define ThrustCudaPolicyPerThread ::thrust::cuda::par.on(cudaStreamPerThread)

template <typename T, typename BinaryOperationType>
__global__ void SumExclusiveScan(const T a, const T b, T result,
                                 BinaryOperationType binary_op) {
  result = binary_op(a, b);
}

template <typename T>
__host__ static typename std::int64_t
ScanExclusivShim(T begin, T end, std::int64_t *output,
                 std::int64_t initialValue) {
  // we have size three so that we can store the origin end value, the
  // new end value, and the sum of those two
  ::thrust::system::cuda::vector<std::int64_t> sum(3);

  // store the current value of the last position array in a separate cuda
  // memory location since the exclusive_scan will overwrite that value
  // once run
  ::thrust::copy_n(ThrustCudaPolicyPerThread, end - 1, 1, sum.begin());

  auto end_o = ::thrust::exclusive_scan(ThrustCudaPolicyPerThread, begin, end,
                                        output, initialValue);

  // Store the new value for the end of the array. This is done because
  // with items such as the transpose array it is unsafe to pass the
  // portal to the SumExclusiveScan
  ::thrust::copy_n(ThrustCudaPolicyPerThread, (end_o - 1), 1, sum.begin() + 1);

  // execute the binaryOp one last time on the device.
  SumExclusiveScan<<<1, 1, 0, cudaStreamPerThread>>>(
      sum[0], sum[1], sum[2], thrust::plus<std::int64_t>{});

  return sum[2];
}

void UseScanExclusivShim() {
  {
    std::int64_t *begin = nullptr, *end = nullptr, *output = nullptr;
    ScanExclusivShim(begin, end, output, 1);
  }

  {
    thrust::counting_iterator<std::int64_t> begin(1024);
    auto end = begin + 42;
    std::int64_t *output = nullptr;
    ScanExclusivShim(begin, end, output, 1);
  }
}
