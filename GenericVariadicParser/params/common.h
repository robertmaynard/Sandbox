#ifndef __params_common_h
#define __params_common_h

#include "limits.h"

#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/sequence/intrinsic/at_c.hpp>
#include <boost/fusion/sequence/intrinsic/size.hpp>

namespace params
{

  namespace fusion = ::boost::fusion;
  using fusion::vector;
  using fusion::at_c;

  //pre GCC 4.7 has a bug with template expansion into
  //non-variadic class template (aka base case).
  //see gcc bug 35722, for the workaround I am using.
  template< template <class...> class T, class... Args>
  struct Join { typedef T<Args...> type; };

  namespace detail
  {
  //determine at compile time the length of a fusion sequence.
  template<class Sequence>
  struct num_elements
    {
    enum{value=boost::fusion::result_of::size<Sequence>::type::value};
    };
  }

}

#endif
