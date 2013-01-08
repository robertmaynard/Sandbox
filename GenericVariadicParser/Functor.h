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
    typedef typename detail::Join<detail::forEach,Args...>::type ForEachType;
    ForEachType()(detail::make_bitwiseLShift(std::cout),args...);
    }
};

class NewFunctorType
{
public:
  //signature that will be used to verify we get the correct number of values
  template<class... Args>
  void operator()(Args... args) const
    {
    typedef typename detail::Join<detail::forEach,Args...>::type ForEachType;
    ForEachType()(detail::make_bitwiseLShift(std::cout),args...);
    }
};

}

#endif
