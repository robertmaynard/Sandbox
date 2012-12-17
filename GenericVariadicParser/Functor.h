#ifndef __functor_
#define __functor_

#include <iostream>

namespace functor {

class MyFunctor
{
public:
  //signature that will be used to verify we get the correct number of values
  template<class... Args>
  void operator()(Args... args) const
    {
    std::cout << "calling MyFunctor variadic version";
    }
};

class NewFunctorType
{
public:
  //signature that will be used to verify we get the correct number of values
  template<class... Args>
  void operator()(Args... args) const
    {
    std::cout << "calling new functor variadic version";
    }
};

}

#endif
