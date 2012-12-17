#ifndef __functor_
#define __functor_

#include <iostream>
#include "Helpers.h"

namespace functor {

class MyFunctor
{
public:
  //signature that will be used to verify we get the correct number of values
  template<class... Args>
  void operator()(Args... args) const
    {
    std::cout << "calling MyFunctor variadic version: ";
    detail::forEach<Args...>()(detail::make_bitwiseLShift(std::cout),args...);
    }

  void operator()(int a) const
    {
    std::cout << "calling MyFunctor int version";
    }
};

class NewFunctorType
{
public:
  //signature that will be used to verify we get the correct number of values
  template<class... Args>
  void operator()(Args... args) const
    {
    std::cout << "calling new functor variadic version: ";

    detail::forEach<Args...>()(detail::make_bitwiseLShift(std::cout),args...);
    }
  void operator()(int a) const
    {
    std::cout << "calling new functor int version";
    }
};

}

#endif
