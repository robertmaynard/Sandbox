#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <vector>

template <class... T> struct list {};
template<class T> struct type_ { using type = T; };

namespace detail
{
template<class, class>
  struct handle_odd_length_lists;
  template<template<class...> class List, class... Ts, class... Us>
  struct handle_odd_length_lists<List<Ts...>, List<Us...>>
  {
    using type = List<Ts..., Ts..., Us...>;
  };

  template<class T, template<class...> class List, std::size_t N>
  struct filled_list_impl
  : handle_odd_length_lists<
    typename filled_list_impl<T, List, N/2>::type,
    typename filled_list_impl<T, List, N%2>::type
    >
  {};

  template<class T, template<class...> class List>
  struct filled_list_impl<T, List, 1>
  {
    using type = List<T>;
  };
  template<class T, template<class...> class List>
  struct filled_list_impl<T, List, 0>
  {
    using type = List<>;
  };
}

template<std::size_t N, template<class...> class List = list>
using void_list = typename detail::filled_list_impl<const void*, List, N>::type;

namespace detail
{
  template<class T> struct element_at;

  template<class... Ts>
  struct element_at<list<Ts...>>
  {
    template<class T> type_<T> static at(Ts..., type_<T>*, ...);
  };

  template<std::size_t N, class Seq> struct at_impl;

  template<std::size_t N, template<typename...> class L, class... Ts>
  struct at_impl<N,L<Ts...>>
    : decltype(element_at<void_list<N>>::at(static_cast<type_<Ts>*>(nullptr)...))
    {
    };
}

template <class L, std::size_t Index>
using at_c = typename detail::at_impl<Index, L>::type;


#include "checker.h"
int main(int, char*[])
{
  checker();
  return 0;
}
