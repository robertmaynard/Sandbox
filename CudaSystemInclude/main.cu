#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <vector>

namespace {


template<typename T> struct printf_functor
{
  __device__
  void operator()(T x) const
  {
    printf("%f\n", static_cast<float>(x));
  }
};

template<>
struct printf_functor<int>
{
  __device__
  void operator()(int x) const
  {
    printf("%d\n", x);
  }
};

}

template<typename T>
void thrust_print( std::vector<T>& data)
{
  thrust::device_vector<T> vec(data);
  thrust::for_each(thrust::device, vec.begin(), vec.end(), printf_functor<T>());
}

int main(int argc, char** argv)
{
  std::vector<int> int_data(3); int_data[0] = 0; int_data[1] = 1; int_data[2] = 2;
  thrust_print(int_data);


  std::vector<float> float_data(3); float_data[0] = 42.0f; float_data[1] = 42.24f; float_data[2] = 3.14f;
  thrust_print(float_data);


  return 0;
}
