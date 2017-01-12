#include <template.h>
TemplateTester::TemplateTester()
{

}

TemplateTester::~TemplateTester()
{
}

template<typename T>
void
TemplateTester::DoWith(T &t)
{
  t =t*2;
}

float
TemplateTester::Do()
{
  float type1 = 1.1f;
  int type2 = 11;
  DoWith(type1);
  DoWith(type2);

  return type1+type2;
}

/*
* Why we need to do explicit template instantiation
* 14 Templates
* 6 A function template, member function of a class template, variable template,
* or static data member of a class template shall be defined in every translation
* unit in which it is implicitly instantiated (14.7.1) unless the corresponding
* specialization is explicitly instantiated (14.7.2) in some translation unit;
* no diagnostic is required.
*
* So basically because we are doing an implicit instantiation ( lines 23 && 24 )
* the symbols are exported as internal only as they could be defined differently
* for other translation units. If we want those template methods shared across
* multiple translation units or libraries we need to do an explicit template
* instantiation.
*
*/
template void TemplateTester::DoWith<int>(int&);