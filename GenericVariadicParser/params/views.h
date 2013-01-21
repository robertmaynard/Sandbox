#ifndef BOOST_PP_IS_ITERATING

#ifndef __params_detail_views_h
#define __params_detail_views_h

#include "limits"
#include <boost/mpl/vector_c.hpp>
#include <boost/fusion/view/nview.hpp>


///////////////////////////////////////////////////////////////////////////////
// Variadic Implementation
///////////////////////////////////////////////////////////////////////////////
#if defined(VARIADIC_SUPPORT)

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
  } //namespace params::detail

  template<class Sequence, int Size>
  struct make_lview
  {
    typedef typename ::params::detail::make_nview<Sequence,0,Size>::type type;
  };

  template<class Sequence, int Size>
  struct make_rview
  {
  private:
    enum{ len = ::params::detail::num_elements<Sequence>::value,
          pos = len - Size };
  public:
    typedef typename ::params::detail::make_nview<Sequence,pos,len>::type type;
  };


} //namespace params

#else //VARIADIC_SUPPORT
# define _lview_index__(n) ##n
# define _rview_index__(n) BOOST_PP_SUB(Length,n)
# define BOOST_PP_ITERATION_PARAMS_1 (3, (2, FUSION_MAX_VECTOR_SIZE,"params/views.h"))

namespace params
{
  namespace detail {
    template<class Sequence, int N> struct make_lview{};
    template<class Sequence, int Length, int N> struct make_rview{};
  } //namespace params::detail

  //iterate to include the implementation of lview and rview
  #include BOOST_PP_ITERATE()

  template<class Sequence, int Size>
  struct make_lview
  {
    typedef typename ::params::detail::make_lview<Sequence,Size>::type type;
  };

  template<class Sequence, int Size>
  struct make_rview
  {
     enum{ len = ::params::detail::num_elements<Sequence>::value };
    typedef typename ::params::detail::make_rview<Sequence,len, Size>::type type;
  };

} //namespace params

#endif //VARIADIC_SUPPORT
#endif //__params_detail_views_h

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)
namespace params
{
namespace detail
{
  template<class Sequence>
  struct make_lview<Sequence, _dax_pp_sizeof___T >
  {
    typedef boost::mpl::vector_c<int,
                                  _dax_pp_enum___( _lview_index__) > itemIndices;
    typdef ::boost::fusion::nview<Sequence,itemIndices> type;
    }
  };

  template<class Sequence, int Length>
  struct make_rview<Sequence, Length, _dax_pp_sizeof___T >
  {
    typedef boost::mpl::vector_c<int,
                                  _dax_pp_enum___( _rview_index__) > itemIndices;
    typdef ::boost::fusion::nview<Sequence,itemIndices> type;
    }
  };

}}  //namespace params::detail

#endif // BOOST_PP_IS_ITERATING
