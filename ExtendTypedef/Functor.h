
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
  typedef arg::placeholders::_5 _5;
  typedef arg::placeholders::_6 _6;
  typedef arg::placeholders::_7 _7;
  typedef arg::placeholders::_8 _8;
  typedef arg::placeholders::_9 _9;

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

class DerivedLotsOfArgs: public  Base
{
public:
  typedef int ControlSignature(Field,Field,Field,Field,Field,Field,Field,Field);
  typedef _8 ExecutionSignature(Replace, _1, _2, _3, _4, _5, _6, _7);
};


}

#endif
