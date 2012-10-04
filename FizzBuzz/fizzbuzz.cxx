
#include <iostream>

template<bool Value> struct printIfTrue
{
  template<typename T>
  void operator()(const T&){}
};

template<> struct printIfTrue<true>
{
  template<typename T>
  void operator()(const T& msg)
  {
    std::cout << msg;
  }
};

template<int Current>
struct findFizzBuzz
{
  void operator()()
    {
    printIfTrue<Current%3==0>()("Fizz");
    printIfTrue<Current%5==0>()("Buzz");
    printIfTrue<Current%3!=0&&Current%5!=0>()(Current);
    std::cout << std::endl;
    findFizzBuzz<Current+1>()();
    }
};

template<>
struct findFizzBuzz<100>
{
  void operator()()
    {
    }
};

int main(int argc, char* argv[])
  {
  findFizzBuzz<1>()();
  return 0;
  }
