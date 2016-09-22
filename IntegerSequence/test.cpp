
#include <vector>
#include <iostream>

// #define USE_UNROLLED

// integer sequence -----------------------------
template<size_t...> struct IntegerSequence{};

#ifdef USE_UNROLLED
template<size_t N, size_t... Is>
struct MakeIntegerSequence : MakeIntegerSequence<N-4, N-3, N-2, N-1, N, Is...>
{ };

template<>
struct MakeIntegerSequence<0>
{ using type = IntegerSequence<0>; };

template<>
struct MakeIntegerSequence<1>
{ using type = IntegerSequence<0,1>; };

template<>
struct MakeIntegerSequence<2>
{ using type = IntegerSequence<0,1,2>; };

template<>
struct MakeIntegerSequence<3>
{ using type = IntegerSequence<0,1,2,3>; };

template<size_t... Ts>
struct MakeIntegerSequence<0,1,2,3,Ts...>
{ using type = IntegerSequence<0,1,2,3,Ts...>; };

template<size_t... Ts>
struct MakeIntegerSequence<-1UL,0,1,2,Ts...>
{ using type = IntegerSequence<0,1,2,Ts...>; };

template<size_t... Ts>
struct MakeIntegerSequence<-2UL,-1UL,0,1,Ts...>
{ using type = IntegerSequence<0,1,Ts...>; };

template<size_t... Ts>
struct MakeIntegerSequence<-3UL,-2UL,-1UL,0,Ts...>
{ using type = IntegerSequence<0,Ts...>; };

#else

template<size_t N, size_t... Is>
struct MakeIntegerSequence :
  MakeIntegerSequence< N-1,
                       N,
                       Is...>
{
};

template<size_t... Ts>
struct MakeIntegerSequence<0,Ts...>
{ using type = IntegerSequence<0,Ts...>; };


#endif

template<size_t... Ts>
void for_each(IntegerSequence<Ts...>)
{

  std::cout << sizeof...(Ts)-1 << std::endl;
}


int main()
{
  using zero = MakeIntegerSequence<0>::type;
  using two = MakeIntegerSequence<2>::type;
  using four = MakeIntegerSequence<4>::type;
  using thirty_two = MakeIntegerSequence<32>::type;
  using thirty_three = MakeIntegerSequence<33>::type;
  using thirty_four = MakeIntegerSequence<34>::type;
  using thirty_five = MakeIntegerSequence<35>::type;
  using five_twelve = MakeIntegerSequence<512>::type;
  using ten_twenty_four = MakeIntegerSequence<1024>::type;

  for_each(zero());
  for_each(two());
  for_each(four());
  for_each(thirty_two());
  for_each(thirty_three());
  for_each(five_twelve());
  for_each(ten_twenty_four());

  return 0;
}
