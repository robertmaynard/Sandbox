
#include <vector>
#include <iostream>

// integer sequence -----------------------------
template<size_t...> struct IntegerSequence{};

namespace detail {

template<size_t N, size_t... Is>
struct MakeSeq : MakeSeq<N-8, N-7, N-6, N-5, N-4, N-3, N-2, N-1, N, Is...>
{ };

template<size_t... Is>
struct MakeSeq<0,1,2,3,4,5,6,7,Is...>
{ using type = IntegerSequence<0,1,2,3,4,5,6,7,Is...>; };

template<size_t Mod, size_t N>
struct PreMakeSeq : MakeSeq<N-7, N-6, N-5, N-4, N-3, N-2, N-1, N> {};

template<size_t N> //specialization for value +1 to divisible by 8
struct PreMakeSeq<1,N> : MakeSeq<N> {};

template<size_t N> //specialization for value +2 to divisible by 8
struct PreMakeSeq<2,N> : MakeSeq<N-1,N> {};

template<size_t N> //specialization for value +3 to divisible by 8
struct PreMakeSeq<3,N> : MakeSeq<N-2,N-1,N> {};

template<size_t N> //specialization for value +4 to divisible by 8
struct PreMakeSeq<4,N> : MakeSeq<N-3,N-2,N-1,N> {};

template<size_t N> //specialization for value +5 to divisible by 8
struct PreMakeSeq<5,N> : MakeSeq<N-4,N-3,N-2,N-1,N> {};

template<size_t N> //specialization for value +6 to divisible by 8
struct PreMakeSeq<6,N> : MakeSeq<N-5,N-4,N-3,N-2,N-1,N> {};

template<size_t N> //specialization for value +7 to divisible by 8
struct PreMakeSeq<7,N> : MakeSeq<N-6,N-5,N-4,N-3,N-2,N-1,N> {};


template<> //specialization for 8
struct PreMakeSeq<0,7> { using type = IntegerSequence<0,1,2,3,4,5,6,7>; };

template<> //specialization for 7
struct PreMakeSeq<7,6> { using type = IntegerSequence<0,1,2,3,4,5,6>; };

template<> //specialization for 6
struct PreMakeSeq<6,5> { using type = IntegerSequence<0,1,2,3,4,5>; };

template<> //specialization for 5
struct PreMakeSeq<5,4> { using type = IntegerSequence<0,1,2,3,4>; };

template<> //specialization for 4
struct PreMakeSeq<4,3> { using type = IntegerSequence<0,1,2,3>; };

template<> //specialization for 3
struct PreMakeSeq<3,2> { using type = IntegerSequence<0,1,2>; };

template<> //specialization for 2
struct PreMakeSeq<2,1> { using type = IntegerSequence<0,1>; };

template<> //specialization for 1
struct PreMakeSeq<1,0> { using type = IntegerSequence<0>; };

template<> //specialization for 0
struct PreMakeSeq<0,-1UL> { using type = IntegerSequence<>; };



} //end detail

template<std::size_t N>
struct MakeIntegerSequence : detail::PreMakeSeq<N%8,N-1> { };

// template<>
// struct MakeIntegerSequence<0> : detail::MakeSeq<0> { };

template<size_t... Ts>
void for_each(IntegerSequence<Ts...>)
{

  std::cout << sizeof...(Ts) << std::endl;
}


int main()
{
  using zero = MakeIntegerSequence<0>::type;
  using two = MakeIntegerSequence<2>::type;
  using four = MakeIntegerSequence<4>::type;
  using five = MakeIntegerSequence<5>::type;
  using six = MakeIntegerSequence<6>::type;
  using thirty_two = MakeIntegerSequence<32>::type;
  using thirty_three = MakeIntegerSequence<33>::type;
  using thirty_four = MakeIntegerSequence<34>::type;
  using thirty_five = MakeIntegerSequence<35>::type;
  using five_twelve = MakeIntegerSequence<512>::type;
  using ten_twenty_four = MakeIntegerSequence<1024>::type;
  using twelve_eighty = MakeIntegerSequence<1280>::type;
  using twelve_eighty_one = MakeIntegerSequence<1281>::type;

  for_each(zero());
  for_each(two());
  for_each(four());
  for_each(five());
  for_each(six());
  for_each(thirty_two());
  for_each(thirty_three());
  for_each(thirty_four());
  for_each(thirty_five());
  for_each(five_twelve());
  for_each(ten_twenty_four());
  for_each(twelve_eighty());
  for_each(twelve_eighty_one());

  return 0;
}
