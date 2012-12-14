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
//strip N element off args off
template< template<class ...> class Factory, int N, class T, class ...OtherArgs>
struct strip
{
  typedef typename strip<Factory,N-1,OtherArgs...>::type type;

  type operator()(T t, OtherArgs... args) const
  {
    return strip<Factory,N-1,OtherArgs...>()(args...);
  }
};

template<template<class ...> class Factory, class T, class ...OtherArgs>
struct strip<Factory, 0,T, OtherArgs...>
{
  typedef Factory<T, OtherArgs...> type;

  type operator()(T t, OtherArgs... args) const
  {
    return type(t,args...);
  }
};


// //empty default implementation, we need it to to do specializations
// template <typename...> struct join;

// //join is basically tuple_cat but with only two parameters, looks like
// //it may be needed a I can't find the respective function in boost
// template <
//           int ...Indices1,
//           int ...Indices2,
//           typename ...Args1,
//           typename ...Args2
//           >
// struct join<detail::sequence<Indices1...>,
//             detail::sequence<Indices2...>,
//             std::tr1::tuple<Args1...>,
//             std::tr1::tuple<Args2...> >
// {
// private:
//   //the entire problem is that a raw tuples arguments include std::tr1::_NullClass
//   //which can't be joined together. So we need to use subset to strip all those
//   //out of each parameter pack and than use convert_to_joined_tuple to
//   //combine everything back together again.

//   //find the lengths of both tuples, i need a better method to do this
//   enum { lenArg1 = std::tr1::tuple_size< std::tr1::tuple<Args1...> >::value,
//          lenArg2 = std::tr1::tuple_size< std::tr1::tuple<Args2...> >::value};

//   typedef params::subset<lenArg1,Args1...> subset1;
//   typedef params::subset<lenArg2,Args2...> subset2;
//   typedef detail::convert_two_to_tuple<subset1::template apply,
//                                        subset2::template apply> FindJoinedType;

// public:
//   typedef typename FindJoinedType::type type;


//   type operator()(detail::sequence<Indices1...>,
//                   detail::sequence<Indices2...>,
//                   std::tr1::tuple<Args1...> first,
//                   std::tr1::tuple<Args2...> second) const
//   {
//     return type(std::tr1::get<Indices1>(first)...,
//                 std::tr1::get<Indices2>(second)...);
//   };

// };

// //join is basically tuple_cat but with only two parameters, looks like
// //it may be needed a I can't find the respective function in boost
// template <typename ...Args1,
//           typename ...Args2>
// struct join<std::tr1::tuple<Args1...>,std::tr1::tuple<Args2...> >
// {
// private:
//     //find the lengths of both tuples, i need a better method to do this
//     enum { lenArg1 = std::tr1::tuple_size< std::tr1::tuple<Args1...> >::value,
//          lenArg2 = std::tr1::tuple_size< std::tr1::tuple<Args2...> >::value};

//     //determine the types of a sequence that matches the length of each argument.
//     //To pass multiple parameter packs to a method they have to be wrapped as
//     //template arguments to a struct/class.
//     typedef typename detail::generate_sequence<lenArg1>::type Arg1SeqType;
//     typedef typename detail::generate_sequence<lenArg2>::type Arg2SeqType;

//     //construct easier typedefs for each tuple arg
//     typedef std::tr1::tuple<Args1...> FirstTuple;
//     typedef std::tr1::tuple<Args2...> SecondTuple;

//     //determine the signature for the actual join struct
//     typedef params::join<Arg1SeqType,Arg2SeqType,FirstTuple,SecondTuple> IndexedJoin;

// public:

//   typedef typename IndexedJoin::type type;

//   type operator()(std::tr1::tuple<Args1...> first,
//                   std::tr1::tuple<Args2...> second) const
//   {
//     //return a joined object
//     return IndexedJoin()(Arg1SeqType(),Arg2SeqType(),first,second);
//   }
// };





};

#endif
