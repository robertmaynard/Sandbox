#include "DerivedParser.h"
#include "DefaultParser.h"

#include "Functor.h"
#include <boost/timer/timer.hpp>

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

template<typename Functor, typename ...Values>
void TryDirectCall(Functor f, Values... v)
{
  f (v...);
  std::cout << std::endl;
}

struct test_class
{
  test_class():id(next_id++){}
  test_class(const test_class& copy_from_me):id(next_id++){}
  friend std::ostream& operator<<(std::ostream& os, const test_class& tc);

  int id;
private:
  static int next_id;
};

std::ostream& operator<<(std::ostream& os, const test_class& tc)
{
    os << "test_class id: " << tc.id;
    return os;
}

int test_class::next_id = 0;


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
