#ifndef __ParameterPacks_h
#define __ParameterPacks_h

#include <algorithm>

#define BOOST_FUSION_INVOKE_PROCEDURE_MAX_ARITY 20
#define FUSION_MAX_VECTOR_SIZE 20


#include <boost/fusion/algorithm/transformation/insert_range.hpp>
#include <boost/fusion/algorithm/transformation/push_back.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/functional/invocation/invoke_procedure.hpp>
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
#include <boost/fusion/view/single_view.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/push_front.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/type_traits/remove_reference.hpp>

//work around a bug in boost::fusion::is_view in that it
//doesn't properly implement the non_fusion_tag as meaning something isn't
//a view
namespace boost { namespace fusion { namespace extension {
  template <>
  struct is_view_impl<non_fusion_tag>
  {
      template <typename Sequence>
      struct apply : mpl::false_ {};
  };
}}}

namespace params  {
namespace detail {

  namespace fusion = ::boost::fusion;

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

  //determine at compile time the length of a fusion sequence.
  template<class Sequence>
  struct num_elements { enum{value=fusion::result_of::size<Sequence>::type::value}; };

  //wrap a functor in a class the fullfills the callable object requirements of boost fusion
  template<class Functor>
  struct functor_wrapper{ };

  template<class I>
  struct is_fusion_sequence
  {
    //deref I as it is an iterator
    typedef typename fusion::result_of::deref<I>::type derefType;

    //the iterator might be to a sequence or view reference, which
    //doesn't compile, so we have to remove the reference
    typedef typename boost::remove_reference<derefType>::type deducedType;

    //check if it is a sequence or view
    typedef typename fusion::traits::is_sequence<deducedType>::type isSequenceType;
    typedef typename fusion::traits::is_view<deducedType>::type isViewType;
    typedef typename boost::mpl::or_< isSequenceType, isViewType >::type IsIterator;
  public:
    typedef IsIterator type;
  };
}
} //namespace params::detail


namespace params
{
  namespace fusion= ::boost::fusion;
  using fusion::vector;
  using fusion::at_c;
  using fusion::at;

  //pre GCC 4.7 has a bug with template expansion into
  //non-variadic class template (aka base case).
  //see gcc bug 35722, for the workaround I am using.
  template< template <class...> class T, class... Args>
  struct Join { typedef T<Args...> type; };


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
    enum {size=params::detail::num_elements<Sequence>::value};
  public:
    typedef typename params::detail::make_nview<Sequence,0,Leading_Number_To_Remove>::type LeadingView;
    typedef typename params::detail::make_nview<Sequence,Leading_Number_To_Remove,size>::type TrailingView;

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
        //insert every element in the sequence contained in item to the end
        //of the Sequence s.
        flatten_impl<Size,Element+1,Functor>()(f,
                   fusion::next(item),
                   fusion::insert_range(s,fusion::end(s),fusion::deref(item)));
      }

      template<class Item, class Sequence>
      void operator()(Functor& f, Item item, Sequence s,
                      typename boost::disable_if<
                      typename params::detail::is_fusion_sequence<Item>::type >::type* = 0) const
      {
        //push the derefence of the item onto the end of sequence s
        flatten_impl<Size,Element+1,Functor>()(f,
                                     fusion::next(item),
                                     fusion::push_back(s,fusion::deref(item)));
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
      fusion::invoke_procedure(f,s);
      }
    };

    //function that starts the flatten recursion when the first item is a sequence
    template< int Size, class Functor, class Item>
    void flatten(Functor& f, Item item,
                 typename boost::enable_if<
                 typename params::detail::is_fusion_sequence<Item>::type >::type* = 0)
    {
      params::detail::flatten_impl<Size,1,Functor>()(f,fusion::next(item),
                                                       fusion::deref(item));
    }

    //function that starts the flatten recursion when the first item isn't a sequence
    template< int Size, class Functor, class Item>
    void flatten(Functor& f, Item item,
                 typename boost::disable_if<
                 typename params::detail::is_fusion_sequence<Item>::type >::type* = 0)
    {
      typedef typename fusion::result_of::deref<Item>::type ItemDeRef;
      typedef fusion::single_view<ItemDeRef> ItemView;
      ItemView iv(fusion::deref(item));
      params::detail::flatten_impl<Size,1,Functor>()(f,fusion::next(item),iv);
    }
  }

  //take an arbitrary class that has a parameter pack and flatten it so
  //that we can call a method with each element of the class
  template< class Functor,
            class ... Args>
  void flatten(Functor& f, Args... args)
  {
    typedef typename ::params::Join< ::params::vector,Args... >::type Sequence;
    Sequence all_args(args...);

    ::params::detail::flatten<
      ::params::detail::num_elements<Sequence>::value,
      Functor>(f,fusion::begin(all_args));
  }
}

#endif
