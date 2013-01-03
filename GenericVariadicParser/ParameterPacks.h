#ifndef __ParameterPacks_h
#define __ParameterPacks_h

#include <algorithm>

#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/iterator/deref.hpp>
#include <boost/fusion/iterator/next.hpp>
#include <boost/fusion/sequence/intrinsic/at_c.hpp>
#include <boost/fusion/sequence/intrinsic/begin.hpp>
#include <boost/fusion/sequence/intrinsic/end.hpp>
#include <boost/fusion/sequence/intrinsic/size.hpp>
#include <boost/fusion/support/is_sequence.hpp>
#include <boost/fusion/support/is_view.hpp>
#include <boost/fusion/view/joint_view.hpp>
#include <boost/fusion/view/nview.hpp>
#include <boost/fusion/functional/invocation/invoke_procedure.hpp>
#include <boost/fusion/view/single_view.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/vector_c.hpp>

#include <boost/type_traits/remove_reference.hpp>

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

  //wrap a functor in a class the fullfills the callable object requirements of boost fusion
  template<class Functor>
  struct functor_wrapper{ };

  template<class I>
  struct is_fusion_sequence
  {
  private:
    //deref I as it is an iterator
    typedef typename boost::fusion::result_of::deref<I>::type derefType;

    //the iterator might be to a sequence or view reference, which
    //doesn't compile, so we have to remove the reference
    typedef typename boost::remove_reference<derefType>::type deducedType;

    //check if it is a sequence or view
    typedef typename boost::fusion::traits::is_sequence<deducedType>::type isSequenceType;
    typedef typename boost::fusion::traits::is_view<deducedType>::type isViewType;
    typedef typename boost::mpl::or_< isSequenceType, isViewType >::type IsIterator;
  public:
    typedef IsIterator type;
  };
}
} //namespace params::detail


namespace params
{
  using boost::fusion::vector;
  using boost::fusion::at_c;

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
    template< int Size, int Element, class Functor>
    struct flatten_impl
    {
      template<class Item, class Sequence>
      void operator()(Functor& f, Item item, Sequence s,
                      typename boost::enable_if<
                      typename params::detail::is_fusion_sequence<Item>::type >::type* = 0) const
      {
        //item is a sequence so we can use joint_view
        typedef typename boost::fusion::result_of::deref<Item>::type ItemDeRef;
        typedef boost::fusion::joint_view<Sequence,ItemDeRef> NewSequenceType;
        NewSequenceType newSeq(s,boost::fusion::deref(item));
        flatten_impl<Size,Element+1,Functor>()(f,boost::fusion::next(item),newSeq);
      }

      template<class Item, class Sequence>
      void operator()(Functor& f, Item item, Sequence s,
                      typename boost::disable_if<
                      typename params::detail::is_fusion_sequence<Item>::type >::type* = 0) const
      {
        //item isn't a sequence so we have to create a single view than
        //use joint view
        typedef boost::fusion::single_view<Item> ItemView;
        typedef boost::fusion::joint_view<Sequence,ItemView> NewSequenceType;
        ItemView iv(item);
        NewSequenceType newSeq(s,iv);
        flatten_impl<Size,Element+1,Functor>()(f,boost::fusion::next(item),newSeq);
      }
    };

    //termination case of the flatten recursive calls
    template< int Size, class Functor>
    struct flatten_impl<Size, Size, Functor>
    {
      template<class Item, class Sequence>
      void operator()(Functor& f, Item, Sequence s) const
      {
        //item is an iterator pointing to end, so everything is in sequence
        boost::fusion::invoke_procedure(f,s);
      }
    };

    //function that starts the flatten recursion when the first item is a sequence
    template< int Size, class Functor, class Item>
    void flatten(Functor& f, Item item,
                 typename boost::enable_if<
                 typename params::detail::is_fusion_sequence<Item>::type >::type* = 0)
    {
      params::detail::flatten_impl<Size,1,Functor>()(f,boost::fusion::next(item),
                                                       boost::fusion::deref(item));
    }

    //function that starts the flatten recursion when the first item isn't a sequence
    template< int Size, class Functor, class Item>
    void flatten(Functor& f, Item item,
                 typename boost::disable_if<
                 typename params::detail::is_fusion_sequence<Item>::type >::type* = 0)
    {
      typedef boost::fusion::single_view<Item> ItemView;
      ItemView iv(item);
      params::detail::flatten_impl<Size,1,Functor>()(f,boost::fusion::next(item),iv);
    }
  }

  //take an arbitrary class that has a parameter pack and flatten it so
  //that we can call a method with each element of the class
  template< class Functor,
            class ... Args>
  void flatten(Functor& f, Args... args)
  {
    typedef ::params::vector<Args...> Sequence;
    Sequence all_args(args...);

    ::params::detail::flatten<
      ::params::detail::num_elements<Sequence>::value,
      Functor>(f,boost::fusion::begin(all_args));
  }
}

#endif
