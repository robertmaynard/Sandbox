#ifndef __DerivedParser_h
#define __DerivedParser_h

#include "ParserBase.h"
#include "Functor.h"

class DerivedParser : public ParserBase<DerivedParser,2>
{
public:
  template<typename Functor, typename Arg1, typename Arg2, typename OtherArgs>
  bool parse(Functor& f, const Arg1& one, Arg2& two, OtherArgs& others) const
    {
    return params::invoke(f,one,two,others,one,two);
    };
};

#endif
