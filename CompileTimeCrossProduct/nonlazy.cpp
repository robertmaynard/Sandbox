
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

template <class R1, class R2>
struct cross_product
{
  using type = brigand::reverse_fold<
    brigand::list<R1, R2>,
    brigand::list<brigand::list<>>,
    brigand::bind<
      brigand::join,
      brigand::bind<
        brigand::transform,
        brigand::_2,
        brigand::defer<brigand::bind<
          brigand::join,
          brigand::bind<
            brigand::transform,
            brigand::parent<brigand::_1>,
            brigand::defer<brigand::bind<
              brigand::list,
              brigand::bind<brigand::push_front, brigand::_1, brigand::parent<brigand::_1>>>>>>>>>>;
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
