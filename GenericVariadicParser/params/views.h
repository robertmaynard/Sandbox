#ifndef __params_detail_views_h
#define __params_detail_views_h

#include <boost/mpl/vector_c.hpp>
#include <boost/fusion/view/nview.hpp>

namespace params  {
namespace detail {

  //pre GCC 4.7 has a bug with template expansion into
  //non-variadic class template (aka base case).
  //see gcc bug 35722, for the workaround I am using.
  template< template <class, long ...> class T, long... Args>
  struct L_Join { typedef T<int,Args...> type; };

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
    typedef typename L_Join<boost::mpl::vector_c,Indices...>::type itemIndices;
    typedef ::boost::fusion::nview<Sequence,itemIndices> type;
    };

}
} //namespace params::detail

#endif
