#include "cuda.h"
#include <iostream>

#include "foo.h"

Foo::Foo()
  {
  int devid=0, num=0;
  cudaGetDeviceCount(&num);
  cudaGetDevice(&devid);

  std::cout << "device count" << num << std::endl;
  std::cout << "device id" << devid << std::endl;
  }
