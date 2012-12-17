#include "DerivedParser.h"
#include "Helpers.h"

#include "../ExtendTypdef/Functor.h"

#include <iostream>


namespace functor
{

class MyFunctor: public  DerivedLotsOfArgs
{
public:
  typedef int ControlSignature(Field,Field,Field,Field,Field,Field,Field,Field);
  typedef _8 ExecutionSignature(Replace, _1, _2, _3, _4, _5, _6, _7);


  //signature that will be used to verify we get the correct number of values
  template<typename Args...>
  void operator()(Args...&& args) const
    {
    std::cout << "calling variadic version";
    }
};

};


}

template<typename Functor, typename ...Values>
void TryDerivedParser(Functor f, Values... v)
{
  DerivedParser derived;
  derived(f,v...);
  std::cout << std::endl;
}

template<typename Functor, typename ...Values>
void TryBaseParser(Functor f, Values... v)
{
  BaseParser base;
  base(f,v...);
  std::cout << std::endl;
}



int main()
{
  TryDerivedParser(functor::MyFunctor(),0,1,2.01,3.14f,4,5,"6",7,'8');
  TryBaseParser(functor::MyFunctor(),0,1,2.01,3.14f,4,5,"6",7,'8');
}
