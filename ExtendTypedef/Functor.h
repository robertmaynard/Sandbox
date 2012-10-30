
#ifndef __functor_
#define __functor_

#include "Arg.h"

namespace functor {

class Base
{
public:
  typedef arg::placeholders::_1 _1;
  typedef arg::placeholders::_2 _2;
  typedef arg::Field Field;
  typedef arg::Replace Replace;
  typedef arg::InsertedArg InsertedArg;

};

class Derived : public  Base
{
public:
  typedef void ControlSignature(Field,Field);
  typedef void ExecutionSignature(_1,_2,Replace);

  template<typename T, typename U>
  void operator()(T const& in, U &out) const
  {
  }
};


}

#endif
