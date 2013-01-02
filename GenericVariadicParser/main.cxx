#include "DerivedParser.h"
#include "DefaultParser.h"


#include "Functor.h"

#include <boost/timer/timer.hpp>

template<typename Functor, typename ...Values>
void TryDerivedParser(Functor f, Values... v)
{
  DerivedParser derived;
  derived(f,v...);
}

template<typename Functor, typename ...Values>
void TryDefaultParser(Functor f, Values... v)
{
  DefaultParser derived;
  derived(f,v...);
}

template<typename Functor, typename ...Values>
void TryDirectCall(Functor f, Values... v)
{
  f (v...);
  std::cout << std::endl;
}


int main()
{
  boost::timer::cpu_timer timer;
  timer.start();

  const int iterations = 10;
  for(int i=0;i<iterations;++i)
    {
    TryDerivedParser(functor::MyFunctor(),0,1,2.01,3.14f,4,5,"6",7,'8');
    }
  double time = (timer.elapsed().wall)/1.0e9;
  std::cout << "Derived, " << time << std::endl;

  timer.start();
  for(int i=0;i<iterations;++i)
    {
    TryDefaultParser(functor::MyFunctor(),0,1,2.01,3.14f,4,5,"6",7,'8');
    }
  time = (timer.elapsed().wall)/1.0e9;
  std::cout << "Default, " << time << std::endl;

  timer.start();
  for(int i=0;i<iterations;++i)
    {
    TryDirectCall(functor::MyFunctor(),0,1,2.01,3.14f,4,5,"6",7,'8');
    }
  time = (timer.elapsed().wall)/1.0e9;
  std::cout << "Direct, " << time << std::endl;
}
