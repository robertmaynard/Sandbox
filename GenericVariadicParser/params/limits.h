#ifndef __params_limits_h
#define __params_limits_h

#include <boost/config.hpp>

#define BOOST_FUSION_INVOKE_PROCEDURE_MAX_ARITY 20
#define FUSION_MAX_VECTOR_SIZE 20

//these might need to go into a different header that is only introspection
#if defined(BOOST_HAS_VARIADIC_TMPL)
# define VARIADIC_SUPPORT
#endif

//only defined VARIADIC_JOIN_REQUIRED for gcc, not clang which sets __GNUC__
#if defined(__GNUC__) && !defined(__clang__)
  //we have gcc and not clang acting as gcc
  #if ((__GNUC__ == 4) && (__GNUC_MINOR__ == 6) && defined(VARIADIC_SUPPORT))
  //enable this variable since we need to work around gcc bug 35722
  //for version pre 4.7
#   define VARIADIC_JOIN_REQUIRED
  #endif
#endif


//setup preprocessor support
#if defined(VARIADIC_SUPPORT)
#  include <boost/preprocessor/arithmetic/dec.hpp>
#  include <boost/preprocessor/iteration/iterate.hpp>
#  include <boost/preprocessor/punctuation/comma_if.hpp>
#  include <boost/preprocessor/repetition/enum_shifted.hpp>
#  include <boost/preprocessor/repetition/enum_shifted_binary_params.hpp>
#  include <boost/preprocessor/repetition/enum_shifted_params.hpp>
#  include <boost/preprocessor/repetition/repeat_from_to.hpp>
#  define __pp_default_class_Args__   BOOST_PP_ENUM_PARAMS_WITH_A_DEFAULT(FUSION_MAX_VECTOR_SIZE, class Args, boost::fusion::void_)
#  define __pp_enum_Args__            BOOST_PP_ENUM_PARAMS(FUSION_MAX_VECTOR_SIZE, Args)
#  define __pp_class_Args__           BOOST_PP_ENUM_SHIFTED_PARAMS(BOOST_PP_ITERATION(), class Args___)
#  define __pp_Args__                 BOOST_PP_ENUM_SHIFTED_PARAMS(BOOST_PP_ITERATION(), Args___)
#  define __pp_Int_Args__             BOOST_PP_ENUM_SHIFTED_PARAMS(BOOST_PP_ITERATION(), int Args___)
#  define __pp_params_Args__(x)       BOOST_PP_ENUM_SHIFTED_BINARY_PARAMS(BOOST_PP_ITERATION(), Args___, x)
#  define __pp_values_Args__(x)       BOOST_PP_ENUM_SHIFTED_PARAMS(BOOST_PP_ITERATION(), x)
#endif

//only need to include join on gcc 4.6
#if defined(VARIADIC_JOIN_REQUIRED)
# include "join.h"
#endif


#endif
