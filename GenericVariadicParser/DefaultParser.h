#ifndef __DefaultParser_h
#define __DefaultParser_h

#include "ParserBase.h"

class DefaultParser : public ParserBase<DefaultParser,0>
{
public:
  template<typename Functor, typename OtherArgs>
  bool parse(Functor& f, const OtherArgs& others) const
    {
    return params::flatten(f,one,two,others);
    };
};

#endif
