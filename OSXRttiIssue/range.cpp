
#include "range.h"

#include <memory>
#include <typeinfo>
#include <iostream>

// ContainerBase::~ContainerBase(){}


__attribute__((visibility("default")))
RangeBase* make_Range(float x, float y)
{
  const std::type_info& my_info = typeid(RangeBase*);
  const std::type_info& r_info = typeid(Range<float>*);

  std::cout << "A RangeBase hash: " << my_info.hash_code() << std::endl;
  std::cout << "A RangeBase name: " << my_info.name() << std::endl;

  std::cout << "A Range<float> hash: " << r_info.hash_code() << std::endl;
  std::cout << "A Range<float> name: " << r_info.name() << std::endl;

  return new Range<float>(x,y);
}

__attribute__((visibility("default")))
bool is_RangeFloat(RangeBase* x, RangeBase* y)
{
  return true;
}
