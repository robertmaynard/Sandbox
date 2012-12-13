#include "DerivedParser.h"

#include <iostream>

template<typename ...Values>
void TryParser(Values... v)
{
  DerivedParser derived;
  std::cout << "calling derived parser." << std::endl;
  derived(std::cout,v...);
  std::cout << std::endl;
}


int main()
{
  TryParser(1,3.0f,"string");
  TryParser(0,1,2,3,4,5,6,7,8,9);
}
