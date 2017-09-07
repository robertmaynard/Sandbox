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

}

template <std::size_t Index, class... Ts> struct at_c;
template <std::size_t Index, class... Ts>
struct at_c<Index, list<Ts...> >
{
  using type = typename detail::at_impl<Index, Ts...>::type;
};


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

  using front_short_type = at_c<0UL, short_list>::type;
  using back_short_type = at_c<3UL,short_list>::type;

  using front_long_type = at_c<0UL,long_list>::type;
  using long_type_10 = at_c<10UL,long_list>::type;
  using long_type_15 = at_c<15UL,long_list>::type;
  using long_type_20 = at_c<20UL,long_list>::type;
  using long_type_25 = at_c<25UL,long_list>::type;
  using long_type_30 = at_c<30UL,long_list>::type;
  using back_long_type = at_c<35UL,long_list>::type;

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
