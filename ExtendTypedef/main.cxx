#include "Functor.h"
#include "Modify.h"

int main()
{
  typedef Modify<functor::Derived>::Arg1Type AType;
  typedef Modify<functor::Derived>::Arg2Type BType;

  AType a=1;
  BType b=1;

}
