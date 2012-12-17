#include "DerivedParser.h"
#include "DefaultParser.h"


#include "Functor.h"


template<typename Functor, typename ...Values>
void TryDerivedParser(Functor f, Values... v)
{
  DerivedParser derived;
  derived(f,v...);
  std::cout << std::endl;
}

template<typename Functor, typename ...Values>
void TryDefaultParser(Functor f, Values... v)
{
  DefaultParser derived;
  derived(f,v...);
  std::cout << std::endl;
}

int main()
{
  TryDerivedParser(functor::MyFunctor(),0,1,2.01,3.14f,4,5,"6",7,'8');
  TryDefaultParser(functor::MyFunctor(),0,1,2.01,3.14f,4,5,"6",7,'8');
}
