#ifndef __DefaultParser_h
#define __DefaultParser_h

#include "ParserBase.h"

class DefaultParser : public ParserBase<DefaultParser,0>
{
public:
  template<typename Functor, typename OtherArgs>
  bool parse(Functor& f, const OtherArgs& others) const
    {
    const int one=1;
    float two = {2.0f};
    return params::flatten(f,one,two,others);
    };
};

#endif
