#ifndef __ParameterPacks_h
#define __ParameterPacks_h

#include <algorithm>

#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/sequence/intrinsic/at_c.hpp>
#include <boost/fusion/sequence/intrinsic/size.hpp>
#include <boost/fusion/support/is_sequence.hpp>
#include <boost/fusion/support/is_view.hpp>
#include <boost/fusion/view/nview.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/vector_c.hpp>

namespace params  {
namespace detail {
  //generate a mpl vector of incrementing values from Start to End
  //which is inclusive at the start, exclusive at end.
  //start must be less than end
  template<int Start, int End, int ...S>
  struct make_mpl_vector { typedef typename make_mpl_vector<Start,End-1, End-1, S...>::type type; };

  //generate a mpl vector of incrementing values from Start to End
  //which is inclusive at the start, exclusive at end.F
  //start must be less than end
  template<int Start, int ...S>
  struct make_mpl_vector<Start,Start, S...> { typedef boost::mpl::vector_c<int,S...> type; };

  //determine at compile time the length of a fusion sequence.
  template<class Sequence>
  struct num_elements { enum{value=boost::fusion::result_of::size<Sequence>::type::value}; };
}
}

namespace params
{
  using boost::fusion::vector;
  using boost::fusion::at_c;
  using boost::fusion::traits::is_sequence;

  //holds a sequence of integers.
  template<int ...>
  struct static_indices { };

  //generate a struct with template args of incrementing values from Start to End
  //which is inclusive at the start, exclusive at end.
  //start must be less than end
  template<int Start, int End, int ...S>
  struct make_indices { typedef typename make_indices<Start,End-1, End-1, S...>::type type; };

  //generate a struct with tempalte args of incrementing values from Start to End
  //which is inclusive at the start, exclusive at end.
  //start must be less than end
  template<int Start, int ...S>
  struct make_indices<Start,Start, S...> { typedef static_indices<S...> type; };

  template<class Sequence, int Leading_Number_To_Remove>
  struct trim
  {
  private:
    //use mpl and fusion to construct to nview objects
    //that will show only a subset of the items
    typedef typename params::detail::make_mpl_vector<
                    0,Leading_Number_To_Remove>::type LeadingVectorIndices;

    typedef typename params::detail::make_mpl_vector<
                    Leading_Number_To_Remove,
                    params::detail::num_elements<Sequence>::value >::type TrailingVectorIndices;

  public:
    typedef boost::fusion::nview<Sequence,LeadingVectorIndices> LeadingView;

    typedef boost::fusion::nview<Sequence,TrailingVectorIndices> TrailingView;

    LeadingView FrontArgs(Sequence& s) const
    {
    return LeadingView(s);
    }

    TrailingView BackArgs(Sequence& s) const
    {
    return TrailingView(s);
    }
  };

  namespace detail
  {
    template< int Size, int Element, class Functor, template<int,class...> class CallBack, int CallBackN>
    struct flatten_sequence
    {
      //we propagate down CallBack and CallBackN so that we can call flatten
      //with the correct number of args left to flatten. We can't pass Callbacks
      //full signature into this call as it hasn't been determined intill we
      //flatten this tuple
      template< class Sequence, class ...OtherArgs>
      void operator()(Functor& functor, Sequence seq, OtherArgs... theRest) const
      {
        //expand the tuple by extracting elements from the front
        //and pushing them to the back of the OtherArgs.
        flatten_sequence<Size-1,Element+1,Functor,CallBack,CallBackN>()(
                       functor, seq, theRest..., ::params::at_c<Element>(seq));
      }

    };

    template<int Element, class Functor, template<int,class...> class CallBack, int CallBackN>
    struct flatten_sequence<1,Element,Functor,CallBack,CallBackN>
    {
      //specialization for the last argument in the tuple. we push back
      //the last element in the tuple and invoke the callback with the new
      //signature with the tuple flattened.
      template<class Sequence, class ...OtherArgs>
      void operator()(Functor& functor, Sequence seq, OtherArgs... theRest) const
      {
        typedef typename boost::fusion::result_of::at_c<Sequence,Element>::type LastElementType;
        typedef CallBack<CallBackN,Functor,OtherArgs...,LastElementType> CallBackType;
        CallBackType()(functor,theRest...,::params::at_c<Element>(seq));
      }
    };

    template< template<int,class,class,class...> class CallBack,
              int CallBackN,
              class Functor,
              class Arg,
              class ...OtherArgs>
    typename boost::enable_if<::params::is_sequence<Arg>, Arg>::type
    flatten_single_arg(Functor& f, Arg seq, OtherArgs... theRest)
    { //we are a sequence/view so we need to flatten this argument into its values.
      enum{len = params::detail::num_elements<Arg>::value };
      flatten_sequence<len,0,Functor,CallBack,CallBackN>()(f,seq,theRest...);
      return seq;
    }

    template< template<int,class,class,class...> class CallBack,
              int CallBackN,
              class Functor,
              class Arg,
              class ...OtherArgs>
    typename boost::disable_if<::params::is_sequence<Arg>, Arg>::type
    flatten_single_arg(Functor f, Arg arg, OtherArgs... theRest)
    { //this is a single argument no reason to flatten it.
      CallBack<CallBackN,Functor,OtherArgs...,Arg>()(f,theRest...,arg);
      return arg;
    }

    template< int N, class Functor, class First, class ...OtherArgs>
    struct flatten
    {
      void operator()(Functor& f, First first, OtherArgs... args) const
      {
      detail::flatten_single_arg<detail::flatten,N-1>(f,first,args...);
      }
    };

    template< class Functor, class First, class ...OtherArgs>
    struct flatten<0, Functor, First, OtherArgs...>
    {
      void operator()(Functor& f, First first, OtherArgs... args) const
      { //remember that we have rotated enough so pass args to functor in current order
        f(first,args...);
      }
    };
  } //namespace detail

  //take an arbitrary class that has a parameter pack and flatten it so
  //that we can call a method with each element of the class
  template< class Functor,
            class ... Args>
  void flatten(Functor& f, Args... args)
  {
    enum{N=sizeof...(Args)};
    ::params::detail::flatten<N,Functor,Args...>()(f,args...);
  }
}

#endif
