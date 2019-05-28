


#include <thrust/system/cuda/memory_resource.h>
#include <thrust/system/cuda/memory.h>

namespace {

template <typename T>
struct allocator
    : thrust::mr::stateless_resource_allocator<
        T,
        ::thrust::system::cuda::memory_resource
    >
{
private:
    typedef thrust::mr::stateless_resource_allocator<
        T,
        ::thrust::system::cuda::memory_resource
    > base;

public:
  template <typename U>
  struct rebind
  {
    typedef allocator<U> other;
  };

  __host__ __device__
  inline allocator() {}

  __host__ __device__
 inline allocator(const allocator & other) : base(other) {}

  template <typename U>
  __host__ __device__
  inline allocator(const allocator<U> & other) : base(other) {}

  __host__ __device__
  inline ~allocator() {}
};    // struct allocator

// So for thrust 1.9.4+ (CUDA 10.1+) the stateless_resource_allocator has a bug
// where it is not marked as __host__ __device__ && __thrust_exec_check_disable__.
// We initially tried to fix this by providing a new partial specialization on cuda::memory_resource
// But that caused downstream runtime crashes in thrust. We tried just
// marking the constructor of allocator also as host only, and still have
// the problem with the warnings are gone, but the runtime crash remains
//Reported in:
//See: https://github.com/thrust/thrust/issues/972
class CMR
{
 allocator<float>  m_allocf;
};

__host__ __device__  void compute(int& input)
{
  CMR t{};
}

}
