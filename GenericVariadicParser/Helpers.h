#ifndef __Helpers_h
#define __Helpers_h

#include <iostream>

namespace detail
{
//pre GCC 4.7 has a bug with template expansion into
//non-variadic class template (aka base case).
//see gcc bug 35722, for the workaround I am using.
template< template <class...> class T, class... Args>
struct Join { typedef T<Args...> type; };

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

//apply a functor to each element in a parameter pack
template<class First, class ...T>
struct forEach
{
  template<typename Functor>
  void operator()(Functor f, First first, T... t) const
  {
    typedef typename detail::Join<detail::forEach,T...>::type ForEachType;
    f(first);
    ForEachType()(f,t...);
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
  typedef typename detail::Join<detail::forEach,T...>::type ForEachType;
  ForEachType()(f,items...);
}

}

#endif
