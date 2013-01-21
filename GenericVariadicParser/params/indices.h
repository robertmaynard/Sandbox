#ifndef BOOST_PP_IS_ITERATING

#ifndef __params_indices_h
#define __params_indices_h

#include "limits"

///////////////////////////////////////////////////////////////////////////////
// Variadic Implementation
///////////////////////////////////////////////////////////////////////////////
#if defined(VARIADIC_SUPPORT)

#include <boost/preprocessor/arithmetic/dec.hpp>
namespace params
{
  //holds a sequence of integers.
  template<int ...>
  struct static_indices { };

  //generate a struct with template args of incrementing values from Start to End
  //which is inclusive at the start, exclusive at end.
  //start must be less than end
  template<int Start, int End, int ...S>
  struct make_indices { typedef typename make_indices<Start,End-1, End-1, S...>::type type; };

  //generate a struct with tempalte args of incrementing values from Start to End
  //which is inclusive at the start, exclusive at end.
  //start must be less than end
  template<int Start, int ...S>
  struct make_indices<Start,Start, S...> { typedef static_indices<S...> type; };

  //special method that is only needed by the default parser when
  //running in c++11 mode
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
# define BOOST_PP_ITERATION_PARAMS_1 (3, (2, FUSION_MAX_VECTOR_SIZE,"params/indices.h"))

namespace params
{
template<class Derived, int N> struct variadic_parse{};
}

#include BOOST_PP_ITERATE()


# undef _unwrap_ref__
#endif //VARIADIC_SUPPORT

#endif //__params_indices_h

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
