#include <iostream>

#ifdef _WIN32
#  define IMPORT __declspec(dllimport)
#else
#  define IMPORT
#endif

IMPORT bool lib_kernel(int x, int& r);


int main(int argc, char** argv)
{
  int res;
  bool ret = lib_kernel(argc, res);
  return 0;
}
