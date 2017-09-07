#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <vector>

template <class... T> struct list {};
template<class T> struct type_ { using type = T; };

namespace detail
{
template<class, class>
  struct join_lists;
  template<template<class...> class List, class... Ts, class... Us>
  struct join_lists<List<Ts...>, List<Us...>>
  {
    using type = List<Ts..., Ts..., Us...>;
  };
  template<class T, template<class...> class List, std::size_t N>
  struct filled_list_impl
  : join_lists<
    typename filled_list_impl<T, List, N/2>::type,
    typename filled_list_impl<T, List, N - N/2*2>::type
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

  template<class T> T extract_type(type_<T>*);

  template<std::size_t N, class Seq> struct at_impl;

  template<std::size_t N, template<typename...> class L, class... Ts>
  struct at_impl<N,L<Ts...>>
    : decltype(element_at<void_list<N>>::at(static_cast<type_<Ts>*>(nullptr)...))
    {
    };
}

template <class L, std::size_t Index>
using at_c = typename detail::at_impl<Index, L>::type;


int main(int, char*[])
{
  using vec_float = std::vector<float>;
  using short_list = list<float, vec_float, vec_float, long>;
  using long_list =
      list<float, vec_float, vec_float, vec_float, float, vec_float, vec_float,
           vec_float, float, double, double, vec_float, float, vec_float,
           vec_float, vec_float, float, vec_float, long, long, float,
           short, short, vec_float, float, vec_float, vec_float,
           vec_float, float, vec_float, int, int, float, vec_float,
           vec_float, vec_float>;

  using front_short_type = at_c<short_list, 0UL>;
  using back_short_type = at_c<short_list, 3UL>;

  using front_long_type = at_c<long_list, 0UL>;
  using long_type_10 = at_c<long_list, 10UL>;
  using long_type_15 = at_c<long_list, 15UL>;
  using long_type_20 = at_c<long_list, 20UL>;
  using long_type_25 = at_c<long_list, 25UL>;
  using long_type_30 = at_c<long_list, 30UL>;
  using back_long_type = at_c<long_list, 35UL>;

  //should print out long type
  std::cout << "short" << std::endl;
  std::cout << typeid(front_short_type{}).name() << std::endl;
  std::cout << typeid(back_short_type{}).name() << std::endl;

  std::cout << "long" << std::endl;
  std::cout << typeid(front_long_type{}).name() << std::endl;
  std::cout << typeid(long_type_10{}).name() << std::endl;
  std::cout << typeid(long_type_15{}).name() << std::endl;
  std::cout << typeid(long_type_20{}).name() << std::endl;
  std::cout << typeid(long_type_25{}).name() << std::endl;
  std::cout << typeid(long_type_30{}).name() << std::endl;
  std::cout << typeid(back_long_type{}).name() << std::endl;
  return 0;
}
