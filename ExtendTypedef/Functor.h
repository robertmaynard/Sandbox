
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
  typedef int ControlSignature(Field,Field);
  typedef int ExecutionSignature(Replace,_1,_2);
};


class DerivedReturn: public  Base
{
public:
  typedef int ControlSignature(Field);
  typedef _1 ExecutionSignature(Replace);
};

}

#endif
