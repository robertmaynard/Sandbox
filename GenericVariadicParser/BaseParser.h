#ifndef __BaseParser_h
#define __BaseParser_h

#include <tr1/tuple>
#include <tr1/utility>
#include <utility>
#include <iostream>

#include "Helpers.h"
#include "ParameterPacks.h"

template<class Derived,int Seperate_Args>
class BaseParser
{
public:
  template<typename Channel, typename... Args>
  bool operator()(Channel& c, Args... args) const
  {

  //the basic operation is to strip N args
  //from the start of the variadic list and pass
  //those in a unique items to Derived class, and than
  //pack the rest in a tuple class
  typedef typename params::strip<std::tr1::tuple,Seperate_Args,Args...> stripper;
  typedef typename stripper::type TrailingTupleType;
  TrailingTupleType tuple = stripper()(args...);

  //forward the arguments to decrease copies
  return static_cast<const Derived*>(this)->parse(c,tuple);
  }
protected:
  template<typename Channel, typename... Args>
  bool defaultParse(Channel& c,std::tr1::tuple<Args...> args) const
  {
    //construct a super simplistic functor that allows us to dump
    //each item to the channel
    detail::bitwiseLShift<Channel> functor(c);
    detail::for_each(functor,args);
    return true;
  }

  template<typename Channel, typename... Args, typename... Arg2>
  bool defaultParse(Channel& c,std::tr1::tuple<Args...> head_args,
                    std::tr1::tuple<Args...> tail_args) const
  {

  //construct a super simplistic functor that allows us to dump
  //each item to the channel
  detail::bitwiseLShift<Channel> functor(c);
  detail::for_each(functor,head_args);
  detail::for_each(functor,tail_args);
  return true;
  }


};

#endif
