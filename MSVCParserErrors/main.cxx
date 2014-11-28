#include <iostream>

//Bug 1: Can't parse function definitions passed as part of a function typedef
struct F {};
typedef void example(F(F,F), F(F,F,F));

//Bug 2: Can't parse function parameters in a multi parameter template
//This one is harder to show without a full example

template<typename T, typename U>
void y(T t, U u) {};

template<typename T, typename U>
struct X {};

typedef void foo<  X< int, X< void y(*)(int, double), void y(*)( F(*)(F), double ) > >;

int main()
{

  return 0;
}

