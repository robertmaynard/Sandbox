#include "ParallelForEach.h"
class MyFunctor1
{
public:
  virtual void operator()(int i) const
  {
    std::cout << "Hello World!\n";
  }
};

class MyFunctor2 : public MyFunctor1
{
public:
  virtual void operator()(int i) const
  {
    std::cout << "Goodbye World!\n";
  }
};

int main()
{
  parallel_for_each(MyFunctor1(), 0, 100);
  parallel_for_each(MyFunctor2(), 0, 100);

  MyFunctor2 t;
  MyFunctor1* derived =  dynamic_cast<MyFunctor2*>(&t);
  parallel_for_each(*derived, 0, 100);
}

