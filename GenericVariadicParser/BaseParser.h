#ifndef __BaseParser_h
#define __BaseParser_h

#include <tr1/tuple>
#include <tr1/utility>
#include <utility>
#include <iostream>

#include "Helpers.h"

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

  typedef std::tr1::tuple<Args...> ArgTupleType;
  ArgTupleType tuple(args...);

  //forward the arguments to decrease copies
  return static_cast<const Derived*>(this)->parse(c,tuple);
  }
protected:
  template<typename Channel, typename... Args>
  bool defaultParse(Channel& c,Args... args) const
  {
    //subtle note, the items in args can be both individual items
    //or tuples that need to be expanded before being passed on.
    //for now we are going to simplify the code by stating only the last
    //argument can be a tuple


    // //extract the last argument as unique item.
    // typedef typename detail::last<Args...> FetchLastArg;
    // typedef typename FetchLastArg::type LastArgType;
    // LastArgType lastArg = FetchLastArg()(args...);

    // //extract everything but the last argument as a std::tuple
    // typedef typename detail::all_but_last<Args...> AllButLast;
    // typedef typename AllButLast::type AllButLastType;
    // AllButLastType before = AllBustLast(args...);

    // //join the last element which could be a tuple with the rest of the
    // //elements. In the future this operation needs to be done on each
    // //item in args so that we can have each arg as a tuple that we join
    // //by expanding each one
    // typedef typename detail::join<AllBustLastType,LastArgType> Joiner;
    // typedef typename Joiner::type JoinerType;
    // JoinerType joined_args = Joiner()(before,lastArg);


    //construct a super simplistic functor that allows us to dump
    //each item to the channel
    detail::bitwiseLShift<Channel> functor(c);


    typename detail::first<Args...>::type tuple = detail::first<Args...>()(args...);
    detail::for_each(functor,tuple);
    return true;
  }


};

#endif
