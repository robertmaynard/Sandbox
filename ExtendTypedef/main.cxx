#include "Functor.h"
#include "Modify.h"

int main()
{
  typedef Modify<functor::Derived,arg::Replace> ModifiedType;

  typedef GetTypes<ModifiedType::ExecutionSignature> Types;

  Types::Arg1Type a;
  Types::Arg2Type b;
}
