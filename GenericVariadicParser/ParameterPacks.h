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

//trim N element off front of the arguments passed in create Factory object
//with the remaining items
template< template<class ...> class Factory, int N, class T, class ...OtherArgs>
struct ltrim
{
  typedef typename ltrim<Factory,N-1,OtherArgs...>::type type;

  type operator()(T t, OtherArgs... args) const
  {
    return ltrim<Factory,N-1,OtherArgs...>()(args...);
  }
};

//trim N element off front of the arguments passed in create Factory object
//with the remaining items
template<template<class ...> class Factory, class T, class ...OtherArgs>
struct ltrim<Factory, 0,T, OtherArgs...>
{
  typedef Factory<T, OtherArgs...> type;

  type operator()(T t, OtherArgs... args) const
  {
    return type(t,args...);
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
struct rtrim
{
  typedef typename truncate<Factory,TruncateSize,ItemsToDrop-1,OtherArgs...,T>::type type;

  type operator()(T t, OtherArgs... args) const
  {
    return truncate<Factory,TruncateSize,ItemsToDrop-1,OtherArgs...,T>()(args...,t);
  }
};

//create a Factory item with only the first N items in it
template<template<class ...> class Factory, int TruncateSize, class T, class ...OtherArgs>
struct rtrim<Factory, TruncateSize, 0, T, OtherArgs...>
{
  enum{M = sizeof...(OtherArgs) - TruncateSize};
  typedef typename ltrim<Factory,M,OtherArgs...,T>::type type;

  type operator()(T t, OtherArgs... args) const
  {
    return ltrim<Factory,M,OtherArgs...,T>()(args...,t);
  }
};

}

//create a Factory item with only the first N items in it, aka rtrim
template< template<class ...> class Factory,
          int TruncateSize,
          class T,
          class ...OtherArgs>
struct rtrim
{
  typedef typename detail::rtrim<Factory,TruncateSize,TruncateSize-1,OtherArgs...,T>::type type;

  type operator()(T t, OtherArgs... args) const
  {
    return detail::rtrim<Factory,TruncateSize,TruncateSize-1,OtherArgs...,T>()(args...,t);
  }
};

//get the N'th item for a parameter pack
template<int N,  class T, class ...Args>
struct get_item
{
  typedef typename get_item<N-1,Args...>::type type;

  type operator()(T t, Args... args) const
  {
    return get_item<N-1,Args...>()(args...);
  }
};

//get the N'th item for a parameter pack
//termination implementation of the recursion
template<class T, class ...Args>
struct get_item<0,T,Args...>
{
  typedef T type;
  type operator()(T t, Args... args) const
    {
    return t;
    }
};

//take an arbitrary class that has a parameter pack and flatten it so
//that we can call a method with each element of the class
template< class Functor,
          class ...Args>
void flatten(Functor& f, Args... args)
{
  f(0);
};




};

#endif
