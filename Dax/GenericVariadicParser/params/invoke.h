#ifndef BOOST_PP_IS_ITERATING

#ifndef __params_invoke_h
#define __params_invoke_h

#include "common.h"
#include "invoke_detail.h"

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
  bool invoke(Functor& f, Args... args)
  {
    typedef typename ::params::vector_type< Args... >::type Sequence;
    Sequence all_args(args...);

    ::params::detail::invoke<
      ::params::detail::num_elements<Sequence>::value,
      Functor>(f,boost::fusion::begin(all_args));

    return true;
  }

}

#else //VARIADIC_SUPPORT
# define BOOST_PP_ITERATION_PARAMS_1 (3, (2, FUSION_MAX_VECTOR_SIZE,"params/invoke.h"))
# include BOOST_PP_ITERATE()
#endif //VARIADIC_SUPPORT


#endif //__params_invoke_h

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

namespace params
{
  //take an arbitrary class that has a parameter pack and flatten it so
  //that we can call a method with each element of the class
  template< class Functor, __pp_class_Args__>
  bool invoke(Functor& f, __pp_params_Args__(args) )
  {
    typedef typename ::params::vector_type< __pp_Args__ >::type Sequence;
    Sequence all_args(__pp_values_Args__(args));

    ::params::detail::invoke<
      ::params::detail::num_elements<Sequence>::value,
      Functor>(f,boost::fusion::begin(all_args));

    return true;
  }

}
#endif //BOOST_PP_IS_ITERATING
