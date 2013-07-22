#include <cmath>
#include <iostream>

int main(int argc, char** argv)
{
  (void)argv;
#ifndef std::isinf
  std::cout << std::isinf(double(1)) << std::endl;
  return ((double*)(&std::isinf<double>))[argc];
#else
  (void)argc;
  return 0;
#endif
}