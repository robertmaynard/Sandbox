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
  typedef Factory<OtherArgs...> type;

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
  typedef typename rtrim<Factory,TruncateSize,ItemsToDrop-1,OtherArgs...,T>::type type;

  type operator()(T t, OtherArgs... args) const
  {
    return rtrim<Factory,TruncateSize,ItemsToDrop-1,OtherArgs...,T>()(args...,t);
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


namespace detail
{

template< int N, class Functor, template<int,class...> class CallBack, int CallBackN>
struct expand_tuple_for_flatten
{

  template<class ...Args, class ...OtherArgs>
  void operator()(Functor& functor,
                  const std::tr1::tuple<Args...>& tuple,
                  OtherArgs... theRest) const
  {
    enum{len = std::tr1::tuple_size< std::tr1::tuple<Args...> >::value };
    expand_tuple_for_flatten<N-1,Functor,CallBack,CallBackN>()(functor,
                                   tuple,
                                   theRest...,
                                   std::tr1::get<len - N>(tuple));
  }

};

template<class Functor, template<int,class...> class CallBack, int CallBackN>
struct expand_tuple_for_flatten<1,Functor,CallBack,CallBackN>
{

  template<class ...Args, class ...OtherArgs>
  void operator()(Functor& functor,
                  const std::tr1::tuple<Args...>& tuple,
                  OtherArgs... theRest) const
  {
    //don't pass tuple too zero since it has been tacked onto the OtherArgs
    enum{len = std::tr1::tuple_size< std::tr1::tuple<Args...> >::value };
    expand_tuple_for_flatten<0,Functor,CallBack,CallBackN>()(
                                   functor,
                                   theRest...,
                                   std::tr1::get<len - 1>(tuple));
  }

};

template<class Functor, template<int,class...> class CallBack, int CallBackN>
struct expand_tuple_for_flatten<0,Functor,CallBack,CallBackN>
{

  template<class ...Args, class ...OtherArgs>
  void operator()(Functor& functor,
                  OtherArgs... theRest) const
  {
    //on step zero we have nothing to extract from the tuple, instead we
    //go back to flattening the rest of the parameter pack
    CallBack<CallBackN,Functor,OtherArgs...>()(functor,theRest...);
  }

};

template< template<int,class,class,class...> class CallBack, int CallBackN, class Functor,
          class ...Args, class ...OtherArgs>
void flatten_single_arg(Functor& f, std::tr1::tuple<Args...> tuple,
                        OtherArgs... theRest)
{
  enum{len = std::tr1::tuple_size< std::tr1::tuple<Args...> >::value };
  expand_tuple_for_flatten<len,Functor,CallBack,CallBackN>()(f,tuple,theRest...);
}

template< template<int,class,class,class...> class CallBack, int CallBackN, class Functor,
          class Arg, class ...OtherArgs>
void flatten_single_arg(Functor f, Arg arg, OtherArgs... theRest)
{
  std::cout << arg << std::endl;
  CallBack<CallBackN,Functor,OtherArgs...,Arg>()(f,theRest...,arg);
}

template< int N,
          class Functor,
          class First,
          class ...OtherArgs>
struct flatten
{

  void operator()(Functor& f, First first, OtherArgs... args)
  {
    detail::flatten_single_arg<detail::flatten,N-1>(f,first,args...);

  }
};

template< class Functor,
          class First,
          class ...OtherArgs>
struct flatten<1, Functor, First, OtherArgs...>
{
  void operator()(Functor& f, First first, OtherArgs... args)
  {
    f(args...,first);
  }

};

}

//take an arbitrary class that has a parameter pack and flatten it so
//that we can call a method with each element of the class
template< class Functor,
          class ... Args>
void flatten(Functor& f, Args... args)
{
  std::tr1::tuple<std::string,int> asdf("2",3);
  detail::flatten<5,Functor,int,
                    std::tr1::tuple<std::string,int>,float,
                    char,std::string>()(f,1,asdf,4.0f,'5',"6");

  // enum{N=sizeof...(Args)};
  //detail::flatten<N,Functor,Args...>()(f,args...);
}




};

#endif
