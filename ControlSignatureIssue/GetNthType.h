
#if !defined(BOOST_PP_IS_ITERATING)

# ifndef __dax__internal__GetNthType_h
# define __dax__internal__GetNthType_h

#include "ParameterPackCxx03.h"

namespace dax { namespace internal {

// Primary template is not defined.
template <unsigned int N, typename TypeSequence> struct GetNthType;

// Specialize for function types of each arity.
template <typename T0> struct GetNthType<0, T0()> { typedef T0 type; };

#  define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, "GetNthType.h"))
#  include BOOST_PP_ITERATE()


}} // namespace dax::internal

#endif //__dax__internal__GetNthType_h
#else // defined(BOOST_PP_IS_ITERATING)

template <typename T0, typename T1 _dax_pp_comma _dax_pp_typename___T> struct GetNthType<0, T0(T1 _dax_pp_comma _dax_pp_T___)> { typedef T0 type; };
template <typename T0, typename T1 _dax_pp_comma _dax_pp_typename___T> struct GetNthType<1, T0(T1 _dax_pp_comma _dax_pp_T___)> { typedef T1 type; };
template <unsigned int N, typename T0, typename T1 _dax_pp_comma _dax_pp_typename___T> struct GetNthType<N, T0(T1 _dax_pp_comma _dax_pp_T___)> { typedef typename GetNthType<N-1,T0(_dax_pp_T___)>::type type; };

#endif // defined(BOOST_PP_IS_ITERATING)
