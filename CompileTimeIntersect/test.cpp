
#include "brigand/brigand.hpp"

template<typename Type, typename List> struct contains
{
  using find_result = brigand::find< List,
                                     std::is_same< brigand::_1, Type> >;
  using size = brigand::size<find_result>;
  static constexpr bool value = (size::value != 0);
};


template <class T, class U, class L>
struct tag_has_type
{
  using type = typename std::conditional<contains<U, L>::value,
                                         brigand::push_back<T,U>,
                                         T>::type;
};

template <class R1, class R2>
struct intersect
{
  using type =
    brigand::fold< R1,
                    brigand::list<>,
                    tag_has_type<brigand::_state, brigand::_element, brigand::pin<R2> >
                   >;
};


int main()
{
  //example of using intersect
  using l1 = brigand::list<int, char, float, double, long, long long, unsigned int, unsigned char, unsigned long>;
  using l2 = brigand::list<int*, char, double*, long long>;

  using cp = intersect<l1,l2>::type;
  cp &cc = "cp";
  return 0;
}
