

#include <iostream>

std::string impl_name();

int main(int, char*[])
{
  std::cout << "Calling implementation: " << impl_name() << std::endl;
  return 0;
}
