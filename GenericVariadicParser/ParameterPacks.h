#ifndef __ParameterPacks_h
#define __ParameterPacks_h

#include <tr1/tuple>
#include <tr1/utility>
#include <utility>
#include <algorithm>

#include "Helpers.h"

namespace params
{

//extract the first element of a parameter packs type or actual item
template <class First, class ...T>
struct first
{
    typedef First type;

    type operator()(First f, T...) const
    {
      return f;
    }
};

//extract the last element of a parameter packs type or actual item
template<class First, class ...T>
struct last
{
  typedef typename last<T...>::type type;

  type operator()(First, T... t) const
  {
    return last<T...>::operator()(t...);
  }

};

//extract the last element of a parameter packs type or actual item
template<class First>
struct last<First>
{
  typedef First type;

  type operator()(First first) const
  {
    return first;
  }

};
//strip N element off front of the arguments passed in and create a Factory type
//with those elements
template< template<class ...> class Factory, int N, class T, class ...OtherArgs>
struct strip
{
  typedef typename strip<Factory,N-1,OtherArgs...>::type type;

  type operator()(T t, OtherArgs... args) const
  {
    return strip<Factory,N-1,OtherArgs...>()(args...);
  }
};

//strip N element off front of the arguments passed in and create a Factory type
//with those elements
template<template<class ...> class Factory, class T, class ...OtherArgs>
struct strip<Factory, 0,T, OtherArgs...>
{
  typedef Factory<T, OtherArgs...> type;

  type operator()(T t, OtherArgs... args) const
  {
    return type(t,args...);
  }
};

//create a new object with Args. Can be used to append or push_front
//new arguments to a already generated tuple.
template< template<class ...> class Factory, class ...OtherArgs>
struct make_new
{
  typedef typename Factory<OtherArgs...>::type type;

  type operator()(OtherArgs... args) const
  {
    return Factory<OtherArgs...>(args...);
  }
};

namespace detail
{
  //create a Factory item with only the first N items in it
template< template<class ...> class Factory,
          int TruncateSize,
          int ItemsToDrop,
          class T,
          class ...OtherArgs>
struct truncate
{
  typedef typename truncate<Factory,TruncateSize,ItemsToDrop-1,OtherArgs...,T>::type type;

  type operator()(T t, OtherArgs... args) const
  {
    return truncate<Factory,TruncateSize,ItemsToDrop-1,OtherArgs...,T>()(args...,t);
  }
};

//create a Factory item with only the first N items in it
template<template<class ...> class Factory, int TruncateSize, class T, class ...OtherArgs>
struct truncate<Factory, TruncateSize, 0, T, OtherArgs...>
{
  enum{M = sizeof...(OtherArgs) - TruncateSize};
  typedef typename strip<Factory,M,OtherArgs...,T>::type type;

  type operator()(T t, OtherArgs... args) const
  {
    return strip<Factory,M,OtherArgs...,T>()(args...,t);
  }
};

}

//create a Factory item with only the first N items in it
template< template<class ...> class Factory,
          int TruncateSize,
          class T,
          class ...OtherArgs>
struct truncate
{
  typedef typename detail::truncate<Factory,TruncateSize,TruncateSize-1,OtherArgs...,T>::type type;

  type operator()(T t, OtherArgs... args) const
  {
    return detail::truncate<Factory,TruncateSize,TruncateSize-1,OtherArgs...,T>()(args...,t);
  }
};


};

#endif
