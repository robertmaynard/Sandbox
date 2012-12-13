#ifndef __Helpers_h
#define __Helpers_h

#include <tr1/tuple>
#include <tr1/utility>
#include <utility>
#include <algorithm>

namespace detail
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

//holds a sequence of integers.
template<int ...>
struct sequence { };

//generate a sequence of incrementing values starting at zero
template<int N, int ...S>
struct generate_sequence : generate_sequence<N-1, N-1, S...>
{

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

template<typename Functor, typename ...T>
void for_each_detail(Functor f, T... items)
{
  detail::forEach<T...>()(f,items...);
}


template<typename Functor, typename ...T, int ...Indices>
void for_each_seq(Functor f, std::tr1::tuple<T...> tuple, detail::sequence<Indices...> s)
{
  detail::for_each_detail(f,std::tr1::get<Indices>(tuple)...);
}

template<typename Functor, typename ...T>
void for_each(Functor f, std::tr1::tuple<T...>& tuple)
{
  //to iterate each item in the tuple we have to convert back from
  //a tuple to a parameter pack
  enum { len = std::tr1::tuple_size< std::tr1::tuple<T...> >::value};
  typedef typename generate_sequence<len>::type SequenceType;
  detail::for_each_seq(f,tuple,SequenceType());
}


//basic functor that applies << to each item that comes in to the stored value
template<typename T>
struct bitwiseLShift
{
  bitwiseLShift(T& t):Var(t){}
  template<typename U> void operator()(U u) const { Var << u; }
private:
  T& Var;
};


}

#endif
