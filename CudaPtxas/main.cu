
#include <cuda.h>
#include <math.h>

#include "ErrorMessageBuffer.h"

// Stringify macros for DAX_ASSERT_EXEC
#define __DAX_ASSERT_EXEC_STRINGIFY_2ND(s) #s
#define __DAX_ASSERT_EXEC_STRINGIFY(s) __DAX_ASSERT_EXEC_STRINGIFY_2ND(s)

/// \def DAX_ASSERT_EXEC(condition, work)
///
/// Asserts that \a condition resolves to true.  If \a condition is false,
/// then an error is raised.  This macro is meant to work in the Dax execution
/// environment and requires the \a work object to raise the error and throw it
/// in the control environment.

#ifndef NDEBUG
#define DAX_ASSERT_EXEC(condition, work) \
  if (!(condition)) \
    ::dax::exec::Assert( \
        condition, \
        __FILE__ ":" __DAX_ASSERT_EXEC_STRINGIFY(__LINE__) ": " \
        "Assert Failed (" #condition ")", \
        work)
#else
#define DAX_ASSERT_EXEC(condition, work)
#endif


namespace dax {
namespace exec {

/// Implements the assert functionality of DAX_ASSERT_EXEC.
///
template<class WorkType>
__device__ void Assert(bool condition, const char *message, WorkType work)
{
  if (condition)
    {
    // Do nothing.
    }
  else
    {
    work.RaiseError(message);
    }
}

}
} // namespace dax::exec



#define MY_ASSERT(condition, message) \
  if (!(condition)) \
    { \
    return \
        __FILE__ ":" __DAX_ASSERT_EXEC_STRINGIFY(__LINE__) ": " message \
        " (" #condition ")"; \
    }

namespace dax {
namespace math {

__host__ __device__  float Log2(float x) {
  return log2f(x);
}


} }

template<class Derived>
struct MathTestFunctor
{
  // The original implementation of these kernels just had the tests in the
  // paren operater as you would expect. However, when I modified the test
  // to work in both the control (host) and execution (device) environments,
  // the two had incompatible error reporting mechanisms.  To get arround this
  // problem, I use the paren overload in a curiously recurring template
  // pattern to call the execution-only raise error method in an execution-only
  // method and macros to throw exceptions only in the control environment.

  __device__
  void operator()(int) const
  {
    // Hopefully the derived class will always return constant strings that do
    // not go out of scope. If we get back garbled error strings, this is
    // probably where it happens.
    const char *message = static_cast<const Derived*>(this)->Run();
    if (message != NULL)
      {
      this->ErrorMessage.RaiseError(message);
      }
  }

  dax::exec::internal::ErrorMessageBuffer ErrorMessage;
  __host__
  void SetErrorMessageBuffer(
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->ErrorMessage = errorMessage;
  }
};


struct TestCompareKernel : public MathTestFunctor<TestCompareKernel>
{
  __device__ __host__ const char *Run() const
  {
    float lhs = dax::math::Log2(3.75);
    float rhs =  dax::math::Log2(2.71828183);
    MY_ASSERT( (rhs != lhs) , "died due to signal 10 Bus Error") ;
    return NULL;
  }
};

template<class Functor>
__global__ void launch(Functor f)
{
  f(1);
}


int main(int, char **)
{
  TestCompareKernel kernel;

  // Schedule on device.
  launch<<<1,1>>>(kernel);

  // Run on host. The return value has the same qualification as mentioned
  // before.
  const char *message = kernel.Run();
  if (message != NULL)
    {
    return 1;
    }

  return 0;
}
