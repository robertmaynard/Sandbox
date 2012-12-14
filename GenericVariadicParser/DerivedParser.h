#ifndef __DerivedParser_h
#define __DerivedParser_h

#include "BaseParser.h"


class DerivedParser : public BaseParser<DerivedParser,2>
{
  friend class BaseParser<DerivedParser,2>;
protected:
  template<typename Channel, typename Arg1, typename Arg2, typename OtherArgs>
  bool parse(Channel& c, const Arg1& one, const Arg2& two,
             const OtherArgs& others) const
    {
    std::cout << "calling derived parser." << std::endl;
    double new_arg = 3.14;

    //is this considered a failure?
    std::tr1::tuple<double,Arg1,Arg2> newArgs(new_arg,one,two);
    return this->defaultParse(c,newArgs,others,newArgs);
    };

};

#endif
