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
  template<typename Functor, typename... Args>
  bool operator()(Functor& f, Args... args) const
  {

  //the basic operation is to strip N args
  //from the start of the variadic list and pass
  //those in a unique items to Derived class, and than
  //pack the rest in a tuple class
  typedef typename params::strip<std::tr1::tuple,Seperate_Args,Args...> stripper;
  typedef typename stripper::type TrailingTupleType;

  //tuple is the trailing parameters
  TrailingTupleType trailingArgs = stripper()(args...);

  typedef typename params::truncate<std::tr1::tuple,Seperate_Args,Args...> truncator;
  typedef typename truncator::type LeadingTupleType;

  LeadingTupleType leadingArgs = truncator()(args...);

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
  bool defaultParse(Functor& f,std::tr1::tuple<Args...> args) const
  {
    params::flatten(f,args);
    return true;
  }

  template<typename Functor,
           typename... Args,
           typename... Args2>
  bool defaultParse(Functor& f,
                    std::tr1::tuple<Args...> head_args,
                    std::tr1::tuple<Args2...> tail_args) const
  {
  params::flatten(f,head_args,tail_args);
  return true;
  }

  template<typename Functor,
           typename... Args,
           typename... Args2,
           typename... Args3>
  bool defaultParse(Functor& f,
                    std::tr1::tuple<Args...> head_args,
                    std::tr1::tuple<Args2...> middle_args,
                    std::tr1::tuple<Args3...> tail_args) const

  {
  params::flatten(f,head_args,middle_args,tail_args);
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
                  std::tr1::tuple<LeadingArgs...> leadingArgs,
                  std::tr1::tuple<TrailingArgs...> trailingArgs) const
  {
    return static_cast<const Derived*>(this)->parse(
            f,
            std::tr1::get<LeadingArgIndices>(leadingArgs)...,
            trailingArgs);
  };

};

#endif
