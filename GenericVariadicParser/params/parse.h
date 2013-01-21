#ifndef BOOST_PP_IS_ITERATING

#ifndef __params_parse_h
#define __params_parse_h

//include all the params headers that we need.
//COMMON HAS TO BE FIRST
#include "common.h"


//COMMON HAS TO BE FIRST
#include "invoke.h"
#include "indices.h"
#include "trim.h"


///////////////////////////////////////////////////////////////////////////////
// Variadic Implementation
///////////////////////////////////////////////////////////////////////////////
#if defined(VARIADIC_SUPPORT)
namespace params
{

  template<class Derived,
           class Base,
           class Functor,
           int... Indices,
           class LeadingArgs,
           class TrailingArgs>
  bool variadic_parse(
                  Base* b,
                  Functor& f,
                  params::static_indices<Indices...>,
                  LeadingArgs leading,
                  TrailingArgs trailing)
  {
  //expand the leading args into each item and pass those plus trailing to
  //the derived parser
   return static_cast<const Derived*>(b)->parse(f,
                     boost::unwrap_ref( params::at_c<Indices>(leading))...,
                     trailing);
  }
}

#else //VARIADIC_SUPPORT
# define _unwrap_ref__(n)  boost::unwrap_ref(params::at_c< BOOST_PP_DEC(n) >( leading ))
# define BOOST_PP_ITERATION_PARAMS_1 (3, (2, FUSION_MAX_VECTOR_SIZE,"params/parse.h"))

namespace params
{
template<class Derived, int N> struct variadic_parse{};
}

#include BOOST_PP_ITERATE()


# undef _unwrap_ref__
#endif //VARIADIC_SUPPORT

#endif //__params_parse_h

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)
namespace params
{
  template<class Derived>
  struct variadic_parse<Derived, _dax_pp_sizeof___T >
  {
    template<class Base, class Functor, class LeadingArgs, class TrailingArgs>
    bool operator()(Base* b, Functor& f, LeadingArgs leading, TrailingArgs trailing)
    {
    return static_cast<const Derived*>(b)->parse(f,
                                            _dax_pp_enum___( _unwrap_ref__ ),
                                            trailing);
    }
  };
}

#endif // BOOST_PP_IS_ITERATING
