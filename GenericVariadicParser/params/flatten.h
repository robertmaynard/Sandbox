#ifndef BOOST_PP_IS_ITERATING

#ifndef __params_flatten_h
#define __params_flatten_h

#include "common.h"
#include "flatten_detail.h"

///////////////////////////////////////////////////////////////////////////////
// Variadic Implementation
///////////////////////////////////////////////////////////////////////////////
#if defined(VARIADIC_SUPPORT)

namespace params
{
  //take an arbitrary class that has a parameter pack and flatten it so
  //that we can call a method with each element of the class
  template< class Functor,
            class ... Args>
  void flatten(Functor& f, Args... args)
  {
    typedef typename ::params::vector_type< Args... >::type Sequence;
    Sequence all_args(args...);

    ::params::detail::flatten<
      ::params::detail::num_elements<Sequence>::value,
      Functor>(f,boost::fusion::begin(all_args));
  }

}

#else //VARIADIC_SUPPORT
# define BOOST_PP_ITERATION_PARAMS_1 (3, (2, FUSION_MAX_VECTOR_SIZE,"params/flatten.h"))
# include BOOST_PP_ITERATE()
#endif //VARIADIC_SUPPORT


#endif //__params_flatten_h

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

namespace params
{
  //take an arbitrary class that has a parameter pack and flatten it so
  //that we can call a method with each element of the class
  template< class Functor, __pp_class_Args__>
  void flatten(Functor& f, __pp_params_Args__(args) )
  {
    typedef typename ::params::vector_type< __pp_Args__ >::type Sequence;
    Sequence all_args(__pp_values_Args__(args));

    ::params::detail::flatten<
      ::params::detail::num_elements<Sequence>::value,
      Functor>(f,boost::fusion::begin(all_args));
  }

}
#endif //BOOST_PP_IS_ITERATING
