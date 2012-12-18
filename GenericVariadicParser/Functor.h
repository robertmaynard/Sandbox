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
    detail::forEach<Args...>()(detail::make_emptyFunctor(std::cout),args...);
    }
};

class NewFunctorType
{
public:
  //signature that will be used to verify we get the correct number of values
  template<class... Args>
  void operator()(Args... args) const
    {
    detail::forEach<Args...>()(detail::make_emptyFunctor(std::cout),args...);
    }
};

}

#endif
