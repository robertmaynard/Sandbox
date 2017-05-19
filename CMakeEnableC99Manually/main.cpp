
#include <iostream>

extern "C"
{
  int func(int);
}

int main(int argc, char* argv[])
{
  (void) argv;
  std::cout << func(argc) << std::endl;
  return 0;
}
