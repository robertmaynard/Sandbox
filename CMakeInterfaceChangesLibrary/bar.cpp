

#include <iostream>

std::string impl_name();

int bar()
{
  std::cout << "Calling implementation: " << impl_name() << std::endl;
  return 0;
}
