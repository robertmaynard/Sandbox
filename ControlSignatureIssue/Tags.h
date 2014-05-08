#if !defined(BOOST_PP_IS_ITERATING)

# ifndef __dax__internal__Tags_h
# define __dax__internal__Tags_h

# include "ParameterPackCxx03.h"

# include <boost/mpl/identity.hpp>
# include <boost/mpl/if.hpp>
# include <boost/mpl/or.hpp>
# include <boost/static_assert.hpp>
# include <boost/type_traits/is_base_and_derived.hpp>

namespace dax { namespace internal {

template <typename T> class Tags;

namespace detail {

template <typename T, bool> struct TagsCheckImpl {};
template <typename T> struct TagsCheckImpl<T, true> { typedef T type; };
template <typename B, typename T> struct TagsCheck: public TagsCheckImpl<T, boost::is_base_and_derived<B,T>::value> {};

template <typename T> struct TagsBase;
template <typename B> struct TagsBase<B()> { typedef B base_type; };

template <typename Tag, typename Tags> struct TagsAddImpl;
template <typename Tags1, typename TagOrTags> class TagsAdd;

template <typename Tags1, typename Tag> class TagsAdd<Tags<Tags1>, Tag>
{
  typedef typename Tags<Tags1>::base_type base_type;
  typedef typename TagsCheck<base_type,Tag>::type tag_type;
public:
  typedef Tags<typename boost::mpl::if_<typename Tags<Tags1>::template Has<Tag>,
                                        boost::mpl::identity< Tags1 >,
                                        TagsAddImpl<Tag,Tags1>
                                        >::type::type> type;
};

template <typename Tags1, typename B> class TagsAdd< Tags<Tags1>, B()>
{
  typedef typename Tags<Tags1>::base_type base_type;
  BOOST_STATIC_ASSERT((boost::mpl::or_<boost::is_same<base_type, B>,
                                       boost::is_base_and_derived<base_type, B> >::value));
public:
  typedef Tags<Tags1> type;
};

#  define _dax_TagsCheck(n) TagsCheck<B,T___##n>::type
#  define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, "Tags.h"))
#  include BOOST_PP_ITERATE()
#  undef _dax_TagsCheck

template <typename Tags1, typename Tags2> class TagsAdd< Tags<Tags1>, Tags<Tags2> >: public TagsAdd<Tags<Tags1>,Tags2> {};

template <typename Tags, typename Tag> struct TagsHas;
template <typename T, typename Tag> struct TagsHas<Tags<T>,Tag>: public boost::is_base_and_derived<Tag, Tags<T> > {};

} // namespace detail

template <typename T> class Tags: public detail::TagsBase<T>
{
public:
  template <typename Tag> struct Has: public detail::TagsHas<Tags<T>, Tag> {};
  template <typename TagOrTags> struct Add: public detail::TagsAdd<Tags<T>, TagOrTags> {};
};

}} // namespace dax::internal

# endif //__dax__internal__Tags_h

#else // defined(BOOST_PP_IS_ITERATING)

template <typename Tag, typename B _dax_pp_comma _dax_pp_typename___T> struct TagsAddImpl<Tag, B(_dax_pp_T___)> { typedef B type(_dax_pp_T___ _dax_pp_comma Tag); };
template <typename Tags1, typename B, typename T1 _dax_pp_comma _dax_pp_typename___T> class TagsAdd< Tags<Tags1>, B(T1 _dax_pp_comma _dax_pp_T___)>: public TagsAdd<typename TagsAdd<Tags<Tags1>,T1>::type, B(_dax_pp_T___)> {};
#if _dax_pp_sizeof___T > 0
template <typename B, _dax_pp_typename___T> struct TagsBase<B(_dax_pp_T___)>: _dax_pp_enum___(_dax_TagsCheck) { typedef B base_type; };
#endif // _dax_pp_sizeof___T > 0

#endif // defined(BOOST_PP_IS_ITERATING)
