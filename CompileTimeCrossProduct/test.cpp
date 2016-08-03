
#include <vector>
#include <iostream>

#include "brigand/brigand.hpp"

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
                   append_product< brigand::_state, brigand::_element, U>
                   >;

  using type = brigand::append<T, P>;
};

template <class R1, class R2>
struct cross_product
{
  using type =
    brigand::fold< R2,
                   brigand::list<>,
                   unroll_cross_product_R2< brigand::_state, brigand::_element, R1>
                   >;
};


int main()
{
  //example of using cross product
  using l1 = brigand::list<int, char, float, double, long, long long, unsigned int, unsigned char, unsigned long>;
  using l2 = brigand::list<int *, char *, float*, double*>;

  using cp = cross_product<l1,l2>::type;
  //uncomment to see the generated type
  // cp &cc = "cp";

  return 0;
}
