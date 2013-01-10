#ifndef __params_join_h
#define __params_join_h


namespace params { namespace detail
{
#if defined(VARIADIC_JOIN_REQUIRED)
  //pre GCC 4.7 has a bug with template expansion into
  //non-variadic class template (aka base case).
  //see gcc bug 35722, for the workaround I am using.
  template< template <class...> class T, class... Args>
  struct join { typedef T<Args...> type; };

  //pre GCC 4.7 has a bug with template expansion into
  //non-variadic class templa te (aka base case).
  //see gcc bug 35722, for the workaround I am using.
  template< template <class, long ...> class T, long... Args>
  struct l_join { typedef T<int,Args...> type; };
#endif

} }

#endif
