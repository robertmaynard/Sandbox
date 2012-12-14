#ifndef __Helpers_h
#define __Helpers_h

#include <iostream>

namespace detail
{

//basic functor that applies << to each item that comes in to the stored value
template<typename T>
struct bitwiseLShift
{
  bitwiseLShift(T& t):Var(t){}
  template<typename U> void operator()(U u) const { Var << u << " "; }
private:
  T& Var;
};

//holds a sequence of integers.
template<int ...>
struct sequence { };

//generate a sequence of incrementing values starting at zero
template<int N, int ...S>
struct generate_sequence
{
  typedef typename generate_sequence<N-1, N-1, S...>::type type;
};

//generate a sequence of incrementing values starting at zero
template<int ...S>
struct generate_sequence<0, S...>
{
  typedef sequence<S...> type;
};


//apply a functor to each element in a parameter pack
template<class First, class ...T>
struct forEach
{
  template<typename Functor>
  void operator()(Functor f, First first, T... t) const
  {
    f(first);
    forEach<T...>()(f,t...);
  }

};


// template < template<template<class ...> class> class T >
// struct convert_to_tuple
// {
//   //call method T with tuple as template parameter and assign that
//   //type to type  use this with subset to create a tuple with a subsets type
//   typedef typename T<std::tr1::tuple>::type type;
// };

// template < template<template<class ...> class> class T,
//            template<template<class ...> class> class U>
// struct convert_two_to_tuple
// {
// private:
//   template<typename ...ArgsToHold> struct ArgHolder{};

//   template<typename ...> struct make_as_tuple;
//   template<typename ...Args1,
//          typename ...Args2>
//   struct make_as_tuple<ArgHolder<Args1...>,
//                        ArgHolder<Args2...> >
//     {
//     typedef std::tr1::tuple<Args1...,Args2...> type;
//     };


//   typedef typename T<ArgHolder>::type set1Args;
//   typedef typename U<ArgHolder>::type set2Args;
// public:

//   //do we have to use another!!! inverter?
//   typedef typename make_as_tuple<set1Args,set2Args>::type type;

// };


//apply a functor to each element in a parameter pack
template<class First>
struct forEach<First>
{
  template<typename Functor>
  void operator()(Functor f, First first) const
  {
    f(first);
  }

};

//applies the functor to each element in a parameter pack
template<typename Functor, typename ...T>
void for_each(Functor f, T... items)
{
  detail::forEach<T...>()(f,items...);
}

//special version of for_each that is a helper to get the length of indicies
//as a parameter type. I can't figure out how to do this step inside for_each specialized
//on tuple
template<typename Functor, typename ...T, int ...Indices>
void for_each(Functor f, std::tr1::tuple<T...> tuple, detail::sequence<Indices...>)
{
  detail::for_each(f,std::tr1::get<Indices>(tuple)...);
}

//function overload that detects tuples being sent to for each
//and expands the tuple elements
template<typename Functor, typename ...T>
void for_each(Functor f, std::tr1::tuple<T...>& tuple)
{
  //to iterate each item in the tuple we have to convert back from
  //a tuple to a parameter pack
  enum { len = std::tr1::tuple_size< std::tr1::tuple<T...> >::value};
  typedef typename detail::generate_sequence<len>::type SequenceType;
  detail::for_each(f,tuple,SequenceType());
}

}

#endif
