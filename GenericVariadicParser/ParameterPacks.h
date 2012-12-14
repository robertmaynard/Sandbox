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

//join is basically tuple_cat but with only two parameters, looks like
//it may be needed a I can't find the respective function in boost
template <typename...> struct join;

template <template <typename...> class Tuple,
          typename ...Args1,
          typename ...Args2>
struct join<Tuple<Args1...>,Tuple<Args2...> >
{
  typedef Tuple<Args1..., Args2...> type;

  void operator()(Tuple<Args1...> first, Tuple<Args2...> second) const
  {
    enum { lenArg1 = std::tr1::tuple_size< std::tr1::tuple<Args1...> >::value,
           lenArg2 = std::tr1::tuple_size< std::tr1::tuple<Args2...> >::value};

  }
};




};

#endif
