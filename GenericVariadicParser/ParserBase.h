#ifndef BOOST_PP_IS_ITERATING

#ifndef __ParserBase_h
#define __ParserBase_h

#include <boost/ref.hpp>
#include "params.h"

///////////////////////////////////////////////////////////////////////////////
// Variadic Implementation
///////////////////////////////////////////////////////////////////////////////
#if defined(VARIADIC_SUPPORT)

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

  typedef typename ::params::vector_type<
          boost::reference_wrapper<Args>... >::type ArgVectorType;
  ArgVectorType all_args(boost::ref(args)...);

  typedef typename params::trim<ArgVectorType,Seperate_Args> trimmer;
  typedef typename trimmer::LeadingView LeadingArgsView;
  typedef typename trimmer::TrailingView TrailingArgsView;
  trimmer t;

  LeadingArgsView leadingArgs = t.FrontArgs(all_args);
  TrailingArgsView trailingArgs = t.BackArgs(all_args);
  return this->call_derived_parse(f,leadingArgs,trailingArgs);
  }
private:
  template<class Functor, class LeadingArgs, class TrailingArgs>
  bool call_derived_parse(
                  Functor& f,
                  LeadingArgs leading,
                  TrailingArgs trailing) const
{
  //call the helper method that calls the derived class
  //have to pass tuple as first item
  typedef typename ::params::make_indices<0,Seperate_Args>::type leadingSeqType;
  return params::variadic_parse<Derived>(this,f,leadingSeqType(),leading,trailing);
  };
};


template<class Derived>
class ParserBase<Derived,0>
{
public:
  template<class Functor, class... Args>
  bool operator()(Functor& f, Args... args) const
  {

  typedef typename ::params::vector_type<
          boost::reference_wrapper<Args>... >::type ArgVectorType;
  ArgVectorType all_args(boost::ref(args)...);
  return static_cast<const Derived*>(this)->parse(f,all_args);
  }
};

#else //VARIADIC_SUPPORT
# define _vector_ref_type__(n)  boost::reference_wrapper<Args##n>
# define _vector_make_ref__(n)  boost::ref(args##n))
# define BOOST_PP_ITERATION_PARAMS_1 (3, (2, FUSION_MAX_VECTOR_SIZE,"ParserBase.h"))
# include BOOST_PP_ITERATE()
# undef __pp_ref_params_Args__
# undef __pp_ref_values_Args__
#endif //VARIADIC_SUPPORT


#endif //__ParserBase_h

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

template<class Derived,int Seperate_Args>
class ParserBase
{

public:
  template<class Functor, __pp_class_Args__>
  bool operator()(Functor& f, __pp_params_Args__(args) ) const
  {
  //the basic operation is to strip N args
  //from the start of the variadic list and pass
  //those in a unique items to Derived class, and than
  //pack the rest in a fusion container

  typedef typename ::params::vector_type<
          _dax_pp_enum___(_vector_ref_type__) >::type ArgVectorType;
  ArgVectorType all_args( _dax_pp_enum___(_vector_make_ref__) );

  typedef typename params::trim<ArgVectorType,Seperate_Args> trimmer;
  typedef typename trimmer::LeadingView LeadingArgsView;
  typedef typename trimmer::TrailingView TrailingArgsView;
  trimmer t;

  LeadingArgsView leadingArgs = t.FrontArgs(all_args);
  TrailingArgsView trailingArgs = t.BackArgs(all_args);
  return this->call_derived_parse(f,leadingArgs,trailingArgs);
  }
private:
  template<class Functor, class LeadingArgs, class TrailingArgs>
  bool call_derived_parse(
                  Functor& f,
                  LeadingArgs leading,
                  TrailingArgs trailing) const
  {
  //call the helper method that calls the derived class
  //have to pass tuple as first item

  //this is currently what I am stuck figuring out using pre-processor
  return params::variadic_parse<Derived,Seperate_Args>()(this,f,leading,trailing);
  };
};


template<class Derived>
class ParserBase<Derived,0>
{
public:
  template<class Functor, __pp_class_Args__>
  bool operator()(Functor& f, __pp_params_Args__(args) ) const
  {

    typedef typename ::params::vector_type<
          _dax_pp_enum___(_vector_ref_type__) >::type ArgVectorType;
  ArgVectorType all_args( _dax_pp_enum___(_vector_make_ref__) );

  return static_cast<const Derived*>(this)->parse(f,all_args);
  }

};

#endif
