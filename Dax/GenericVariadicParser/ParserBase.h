#ifndef __ParserBase_h
#define __ParserBase_h

#include <boost/ref.hpp>
#include "params/parse.h"

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
  typedef typename ::params::make_indices<0,Seperate_Args>::type leadingSeqType;
  return params::variadic_parse<Derived>(this,f,leadingSeqType(),
                                         leadingArgs,trailingArgs);
  }
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
///////////////////////////////////////////////////////////////////////////////
//  Preprocessor repeat macros
///////////////////////////////////////////////////////////////////////////////

# define _vector_ref_type__(n)  boost::reference_wrapper<Args___##n>
# define _vector_make_ref__(n)  boost::ref(args##n)
# define __rr_class_Args__(c)      BOOST_PP_ENUM_SHIFTED_PARAMS(c, class Args___)
# define __rr_params_Args__(c,x)  BOOST_PP_ENUM_SHIFTED_BINARY_PARAMS(c, Args___, x)
# define __rr_enum___(c,x)     BOOST_PP_ENUM_SHIFTED(c, _dax_pp_enum, x)

///////////////////////////////////////////////////////////////////////////////
//  ParserBase Implementation
///////////////////////////////////////////////////////////////////////////////
# define _dax_FunctorImpl(z,count,data)                                       \
template<class Functor, __rr_class_Args__(count) >                            \
  bool operator()(Functor& f, __rr_params_Args__(count,args) ) const          \
  {                                                                           \
  typedef typename ::params::vector_type<                                     \
          __rr_enum___(count,_vector_ref_type__) >::type ArgVectorType;       \
  ArgVectorType all_args( __rr_enum___(count,_vector_make_ref__) );           \
                                                                              \
  typedef typename params::trim<ArgVectorType,Seperate_Args> trimmer;         \
  typedef typename trimmer::LeadingView LeadingArgsView;                      \
  typedef typename trimmer::TrailingView TrailingArgsView;                    \
                                                                              \
  trimmer t;                                                                  \
  LeadingArgsView leadingArgs = t.FrontArgs(all_args);                        \
  TrailingArgsView trailingArgs = t.BackArgs(all_args);                       \
  return params::variadic_parse<Derived,Seperate_Args>()(this,                \
                                                 f,leadingArgs,trailingArgs); \
  }

///////////////////////////////////////////////////////////////////////////////
//  ParserBase no args Implementation
///////////////////////////////////////////////////////////////////////////////
# define _dax_FunctorImpl_Zero(z,count,data)                                  \
  template<class Functor, __rr_class_Args__(count) >                          \
  bool operator()(Functor& f, __rr_params_Args__(count,args) ) const          \
  {                                                                           \
    typedef typename ::params::vector_type<                                   \
          __rr_enum___(count,_vector_ref_type__) >::type ArgVectorType;       \
  ArgVectorType all_args( __rr_enum___(count,_vector_make_ref__) );           \
  return static_cast<const Derived*>(this)->parse(f,all_args);                \
  }


///////////////////////////////////////////////////////////////////////////////
//  Where we fire off the local repetition
///////////////////////////////////////////////////////////////////////////////
template<class Derived,int Seperate_Args>
class ParserBase
{
public:
  BOOST_PP_REPEAT_FROM_TO(2,15,_dax_FunctorImpl,BOOST_PP_EMPTY() )
};

template<class Derived>
class ParserBase<Derived,0>
{
public:
  BOOST_PP_REPEAT_FROM_TO(2,15,_dax_FunctorImpl_Zero,BOOST_PP_EMPTY() )
};

#endif //VARIADIC_SUPPORT
#endif //__ParserBase_h
