
#ifndef __params_common_h
#define __params_common_h

#include "limits.h"

#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/sequence/intrinsic/at_c.hpp>
#include <boost/fusion/sequence/intrinsic/size.hpp>

namespace params
{
  using boost::fusion::at_c;
  using boost::fusion::vector;

  namespace detail
  {
  //determine at compile time the length of a fusion sequence.
  template<class Sequence>
  struct num_elements
    {
    enum{value=::boost::fusion::result_of::size<Sequence>::type::value};
    };
  }

///////////////////////////////////////////////////////////////////////////////
// Variadic Implementation
///////////////////////////////////////////////////////////////////////////////
#if defined(VARIADIC_SUPPORT)
  //helper wrapper around join when it is needed on older compilers
  template<class... Args>
  struct vector_type
  {
#if defined(VARIADIC_JOIN_REQUIRED)
    typedef typename params::detail::join< ::params::vector,Args...>::type type;
#else
    typedef ::params::vector<Args...> type;
#endif //VARIADIC_JOIN_REQUIRED
  };
#else //VARIADIC_SUPPORT
///////////////////////////////////////////////////////////////////////////////
// Include Preprocessor Implementation
///////////////////////////////////////////////////////////////////////////////
  template< __pp_default_class_Args__ >
  struct vector_type
  {
  typedef ::params::vector< __pp_enum_Args__ > type;
  };
#endif
}

#endif //__params_common_h
