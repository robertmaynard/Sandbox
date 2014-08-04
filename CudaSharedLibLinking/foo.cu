#include "cuda.h"
#include "zlib.h"

#include <iostream>

#include "foo.h"

Foo::Foo()
  {
  int devid=0, num=0;
  cudaGetDeviceCount(&num);
  cudaGetDevice(&devid);

  std::cout << "device count" << num << std::endl;
  std::cout << "device id" << devid << std::endl;

  int level=1;
  z_stream strm;
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.opaque = Z_NULL;
  int ret = deflateInit(&strm, level);
  std::cout << "deflateInit ret :" << ret << std::endl;
  }
