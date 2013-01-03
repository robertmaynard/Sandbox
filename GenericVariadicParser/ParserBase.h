#ifndef __BaseParser_h
#define __BaseParser_h

#include "ParameterPacks.h"

template<class Derived,int Seperate_Args>
class ParserBase
{

public:
  template<class Functor, class... Args>
  bool operator()(Functor& f, Args... args) const
  {
    //the basic operation is to strip N args
  //from the start of the variadic list and pass
  //those in a unique items to Derived class, and than
  //pack the rest in a fusion container

  typedef const params::vector<Args...> ArgVectorType;
  ArgVectorType all_args(args...);

  typedef typename params::trim<ArgVectorType,Seperate_Args> trimmer;
  typedef typename trimmer::LeadingView LeadingArgsView;
  typedef typename trimmer::TrailingView TrailingArgsView;
  trimmer t;

  LeadingArgsView leadingArgs = t.FrontArgs(all_args);
  TrailingArgsView trailingArgs = t.BackArgs(all_args);

  //call the helper method that calls the derived class
  //have to pass tuple as first item
  typedef typename ::params::make_indices<0,Seperate_Args>::type leadingSeqType;

  return this->call_derived_parse(f,leadingSeqType(),leadingArgs,trailingArgs);
  }
protected:
  template<class Functor,
           class... Args>
  bool defaultParse(Functor& f,const Args&... args) const
  {
    params::flatten(f,args...);
    return true;
  }

private:

  template<class Functor,
           int... Indices,
           class LeadingArgs,
           class TrailingArgs>
  bool call_derived_parse(
                  Functor& f,
                  params::static_indices<Indices...>,
                  LeadingArgs leading,
                  TrailingArgs trailing) const
  {
  //expand the leading args into each item and pass those plus trailing to
  //the derived parser
  return static_cast<const Derived*>(this)->parse(f,
                    boost::unwrap_ref(params::at_c<Indices>(leading))...,
                    trailing);
  };
};


template<class Derived>
class ParserBase<Derived,0>
{
public:
  template<class Functor, class... Args>
  bool operator()(Functor& f, Args... args) const
  {

  typedef params::vector<Args...> ArgVectorType;
  ArgVectorType all_args(args...);

  return static_cast<const Derived*>(this)->parse(f,all_args);
  }
protected:
  template<class Functor,
           class... Args>
  bool defaultParse(Functor& f, const Args&... args) const
  {
    params::flatten(f,args...);
    return true;
  }

};

#endif
