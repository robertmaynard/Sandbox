
#ifndef __dax__internal__ParameterPackCxx03_h
#define __dax__internal__ParameterPackCxx03_h

// In C++03 use Boost.Preprocessor file iteration to approximate
// template parameter packs.
#  include <boost/preprocessor/arithmetic/dec.hpp>
#  include <boost/preprocessor/iteration/iterate.hpp>
#  include <boost/preprocessor/punctuation/comma_if.hpp>
#  include <boost/preprocessor/repetition/enum_shifted.hpp>
#  include <boost/preprocessor/repetition/enum_shifted_binary_params.hpp>
#  include <boost/preprocessor/repetition/enum_shifted_params.hpp>
#  include <boost/preprocessor/repetition/repeat_from_to.hpp>
#  define _dax_pp_T___            BOOST_PP_ENUM_SHIFTED_PARAMS(BOOST_PP_ITERATION(), T___)
#  define _dax_pp_typename___T    BOOST_PP_ENUM_SHIFTED_PARAMS(BOOST_PP_ITERATION(), typename T___)
#  define _dax_pp_sizeof___T      BOOST_PP_DEC(BOOST_PP_ITERATION())
#  define _dax_pp_comma           BOOST_PP_COMMA_IF(_dax_pp_sizeof___T)
#  define _dax_pp_enum___(x)      BOOST_PP_ENUM_SHIFTED(BOOST_PP_ITERATION(), _dax_pp_enum, x)
#  define _dax_pp_enum(z,n,x)     _dax_pp_enum_(z,n,x)
#  define _dax_pp_enum_(z,n,x)    x(n)
#  define _dax_pp_repeat___(x)    BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_ITERATION(), _dax_pp_repeat, x)
#  define _dax_pp_repeat(z,n,x)   _dax_pp_repeat_(z,n,x)
#  define _dax_pp_repeat_(z,n,x)  x(n)
#  define _dax_pp_params___(x)    BOOST_PP_ENUM_SHIFTED_BINARY_PARAMS(BOOST_PP_ITERATION(), T___, x)
#  define _dax_pp_args___(x)      BOOST_PP_ENUM_SHIFTED_PARAMS(BOOST_PP_ITERATION(), x)

#endif //__dax__internal__ParameterPackCxx03_h
