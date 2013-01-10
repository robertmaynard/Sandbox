#ifndef __params_detail_views_h
#define __params_detail_views_h

//only need to include join on gcc 4.6
#if defined(VARIADIC_JOIN_REQUIRED)
# include "join.h"
#endif

#include <boost/mpl/vector_c.hpp>
#include <boost/fusion/view/nview.hpp>

namespace params  {
namespace detail {

  //generate a mpl vector of incrementing values from Start to End
  //which is inclusive at the start, exclusive at end.
  //start must be less than end
  template<class Sequence, int Start, int End, int ...Indices>
  struct make_nview
    {
    typedef typename make_nview<Sequence,Start,End-1,End-1,Indices...>::type type;
    };

  //generate a mpl vector of incrementing values from Start to End
  //which is inclusive at the start, exclusive at end.F
  //start must be less than end
  template<class Sequence, int Start, int ...Indices>
  struct make_nview<Sequence,Start,Start,Indices...>
    {
#if defined(VARIADIC_JOIN_REQUIRED)
    typedef typename params::detail::l_join<boost::mpl::vector_c,Indices...>::type itemIndices;
#else
    typedef boost::mpl::vector_c<int,Indices...> itemIndices;
#endif
    typedef ::boost::fusion::nview<Sequence,itemIndices> type;
    };

}
} //namespace params::detail

#endif
