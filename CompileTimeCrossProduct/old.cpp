
#include <vector>
#include <iostream>
#include <typeinfo>

#include "brigand/brigand.hpp"

template <typename Functor, typename... Args>
void list_for_each(Functor &&, brigand::list<>, Args &&...) {}

template <typename Functor, typename T1, typename... ArgTypes, typename... Args>
void list_for_each(Functor &&f, brigand::list<T1, ArgTypes...> &&,
                   Args &&... args) {
  f(T1{}, std::forward<Args>(args)...);
  list_for_each(std::forward<Functor>(f), brigand::list<ArgTypes...>{},
                std::forward<Args>(args)...);
}

struct printer {
  template <typename T> void operator()(T) const {
    std::cout << "type: " << typeid(T).name() << std::endl;
  }
};
//-----------------------------------------------------------------------------

template <class T, class U, class R>
struct append_product
{
  using type = brigand::push_back<T, std::pair<U,R> >;
};

template <class T, class U, class R2>
struct unroll_cross_product_R2
{
  using P =
    brigand::fold< R2,
                   brigand::list<>,
                   append_product< brigand::_state, brigand::_element, brigand::pin<U> >
                   >;

  using type = brigand::append<T, P>;
};

template <class R1, class R2>
struct cross_product
{
  using type =
    brigand::fold< R2,
                   brigand::list<>,
                   unroll_cross_product_R2< brigand::_state, brigand::_element, brigand::pin<R1> >
                   >;
};

int main()
{
  //example of using cross product
  using l1 =
      brigand::list<int, char, float, double, long, long long, unsigned int,
                    unsigned char, unsigned long, std::vector<int>,
                    std::vector<char>, std::vector<float>, std::vector<float>>;
  using l2 = brigand::list<int *, char *, float *, double *, long *>;

  using cp = cross_product<l1,l2>::type;

  list_for_each( printer{}, cp{});
  return 0;
}
