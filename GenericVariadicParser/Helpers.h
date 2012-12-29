#ifndef __Helpers_h
#define __Helpers_h

#include <iostream>
#include "boost/tuple/tuple.hpp"

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

template<class F>
bitwiseLShift<F> make_bitwiseLShift(F & f)
{
  return bitwiseLShift<F>(f);
}

//basic functor that applies << to each item that comes in to the stored value
template<typename T>
struct emptyFunctor
{
  emptyFunctor(T& t):Var(t){}
  template<typename U> void operator()(U u) const { int x = static_cast<int>(u); }
  void operator()(const char* c) const { int x = *c; }
  void operator()(std::string& s) const { int x = s.at(1); }
private:
  T& Var;
};

template<class F>
emptyFunctor<F> make_emptyFunctor(F & f)
{
  return emptyFunctor<F>(f);
}

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
template<class  Functor, class ...T>
void for_each(Functor f, T... items)
{
  detail::forEach<T...>()(f,items...);
}

//special version of for_each that is a helper to get the length of indicies
//as a parameter type. I can't figure out how to do this step inside for_each specialized
//on tuple
template<class Functor, class ...T, int ...Indices>
void for_each(Functor f, boost::tuple<T...> tuple, detail::sequence<Indices...>)
{
  detail::for_each(f,boost::get<Indices>(tuple)...);
}

//function overload that detects tuples being sent to for each
//and expands the tuple elements
template<class Functor, class ...T>
void for_each(Functor f, boost::tuple<T...>& tuple)
{
  //to iterate each item in the tuple we have to convert back from
  //a tuple to a parameter pack
  enum { len = boost::tuple_size< boost::tuple<T...> >::value};
  typedef typename detail::generate_sequence<len>::type SequenceType;
  detail::for_each(f,tuple,SequenceType());
}

}

#endif
