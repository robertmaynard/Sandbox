
#include "range.h"

#include <memory>
#include <typeinfo>
#include <iostream>

RangeBase* make_Range(float x, float y);
// RangeBase* make_Range(Complex x, Complex y);
bool is_RangeFloat(RangeBase* x, RangeBase* y);

void check_float()
{
  RangeBase* range = make_Range(1.0f, 42.0f);
  Range<float>* my_range = nullptr;
  my_range = dynamic_cast<Range<float>*>(range);

  const std::type_info& my_info = typeid(range);
  const std::type_info& r_info = typeid(Range<float>*);

  std::cout << "ptr from A range<float> hash: " << my_info.hash_code() << std::endl;
  std::cout << "ptr from A range<float> name: " << my_info.name() << std::endl;

  std::cout << "lib B range<float> hash: " << r_info.hash_code() << std::endl;
  std::cout << "lib B range<float> name: " << r_info.name() << std::endl;
  std::cout << my_range << std::endl;
}

void check_is_float()
{
  RangeBase* range = make_Range(1.0f, 42.0f);
  Range<float> my_range(10.0f, 12.0f);
  std::cout << is_RangeFloat(range,&my_range) << std::endl;
}


int main(int argc, char* argv[])
{
  (void) argc;
  (void) argv;

  check_float();

}
