

#include <iostream>

#include "resource.h"

std::string impl_name();



int main(int, char*[])
{
  std::cout << "Calling implementation 1: " << impl_name() << std::endl;
  std::cout << "Calling implementation 2: " << impl_name2() << std::endl;
  return 0;
}
