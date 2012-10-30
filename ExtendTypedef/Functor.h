
#ifndef __functor_
#define __functor_

#include "Arg.h"

namespace functor {

class Base
{
public:
  typedef arg::placeholders::_1 _1;
  typedef arg::placeholders::_2 _2;
  typedef arg::placeholders::_3 _3;
  typedef arg::placeholders::_4 _4;
  typedef arg::Field Field;
  typedef arg::Replace Replace;
  typedef arg::InsertedArg InsertedArg;

};

class Derived : public  Base
{
public:
  typedef void ControlSignature(Field);
  typedef void ExecutionSignature(_1,Replace);
};


class DerivedTwo : public  Base
{
public:
  typedef void ControlSignature(Field,Field,Field);
  typedef void ExecutionSignature(_1,Replace,_2,_3);
};

}

#endif
