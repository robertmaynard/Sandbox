
//use a different header for this section as it needs
//the boost pre-processor

//next step is to convert the boost mpl types back to a worklet
//signature. To get this to work with all functor we need to use
//boost pre-processor
#if !BOOST_PP_IS_ITERATING
# ifndef __buildSignature_
# define __buildSignature_

#  include <boost/mpl/at.hpp>
#  include <boost/preprocessor/iteration/iterate.hpp>
#  include <boost/preprocessor/repetition/enum_shifted_params.hpp>
#  include <boost/preprocessor/repetition/enum_shifted.hpp>

#  define _arg_enum___(x)      BOOST_PP_ENUM_SHIFTED(BOOST_PP_ITERATION(), _arg_enum_, x)
#  define _arg_enum_(z,n,x)    x(n)
#  define _MPL_ARG_(n) typename boost::mpl::at_c<T,n>::type

namespace detail
  {
  template<int N, typename T> struct BuildSig;

# define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 11, "BuildSignature.h"))
# include BOOST_PP_ITERATE()
  }

template<typename T>
struct BuildSignature
{
  typedef boost::mpl::size<T> Size;

  typedef typename ::detail::BuildSig<Size::value,T>::type type;
};

# endif

#else
  template<typename T> struct BuildSig<BOOST_PP_ITERATION(), T>
  {
    typedef typename boost::mpl::at_c<T,0>::type type(_arg_enum___(_MPL_ARG_));


  };
#endif
