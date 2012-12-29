#ifndef __BaseParser_h
#define __BaseParser_h

#include "boost/tuple/tuple.hpp"
#include <iostream>

#include "ParameterPacks.h"

template<class Derived,int Seperate_Args>
class ParserBase
{

public:
  template<typename Functor, typename... Args>
  bool operator()(Functor& f, Args... args) const
  {

  //the basic operation is to strip N args
  //from the start of the variadic list and pass
  //those in a unique items to Derived class, and than
  //pack the rest in a tuple class
  typedef typename params::ltrim<boost::tuple,Seperate_Args,Args...> ltrimmer;
  typedef typename ltrimmer::type TrailingTupleType;

  //tuple is the trailing parameters
  TrailingTupleType trailingArgs = ltrimmer()(args...);

  typedef typename params::rtrim<boost::tuple,Seperate_Args,Args...> rtrimmer;
  typedef typename rtrimmer::type LeadingTupleType;

  LeadingTupleType leadingArgs = rtrimmer()(args...);

  //create a structure that holds the indicies of arguments that we want
  //to pass in as unique items
  typedef typename detail::generate_sequence<Seperate_Args>::type
          UniqueIndiciesType;

  //call the helper method that calls the derived class
  //have to pass tuple as first item
  return this->call_derived_parse(f,
                                  UniqueIndiciesType(),
                                  leadingArgs,
                                  trailingArgs);
  }
protected:
  template<typename Functor,
           typename... Args>
  bool defaultParse(Functor& f,Args... args) const
  {
    params::flatten(f,args...);
    return true;
  }

private:

  template<typename Functor,
           int... LeadingArgIndices,
           typename... LeadingArgs,
           typename... TrailingArgs>
  bool call_derived_parse(
                  Functor& f,
                  detail::sequence<LeadingArgIndices...>,
                  boost::tuple<LeadingArgs...> leadingArgs,
                  boost::tuple<TrailingArgs...> trailingArgs) const
  {
    return static_cast<const Derived*>(this)->parse(
            f,
            boost::get<LeadingArgIndices>(leadingArgs)...,
            trailingArgs);
  };
};


template<class Derived>
class ParserBase<Derived,0>
{
public:
  template<typename Functor, typename... Args>
  bool operator()(Functor& f, Args... args) const
  {

  typedef typename params::make_new<boost::tuple,Args...> tupleMaker;
  typedef typename tupleMaker::type TupleType;

  //tuple is the trailing parameters. I hope we don't have more than 10 items...
  TupleType tuple = tupleMaker()(args...);

  return static_cast<const Derived*>(this)->parse(f,tuple);
  }
protected:
  template<typename Functor,
           typename... Args>
  bool defaultParse(Functor& f,Args... args) const
  {
    params::flatten(f,args...);
    return true;
  }

};

#endif
