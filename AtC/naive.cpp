#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <vector>

template <class... T> struct list {};

namespace detail
{

template< std::size_t Size, typename T, typename... ArgTypes>
struct at_impl
{
  using type = typename at_impl<Size-1, ArgTypes...>::type;
};

template<typename T, typename... ArgTypes>
struct at_impl<0, T, ArgTypes...>
{
  using type = T;
};

template <std::size_t Index, class... Ts> struct at_c;
template <std::size_t Index, class... Ts>
struct at_c<Index, list<Ts...> >
{
  using type = typename detail::at_impl<Index, Ts...>::type;
};

}

template <class L, std::size_t Index>
using at_c = typename detail::at_c<Index, L>::type;


#include "checker.h"
int main(int, char*[])
{
  checker();
  return 0;
}
