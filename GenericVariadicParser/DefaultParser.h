#ifndef __DefaultParser_h
#define __DefaultParser_h

#include "ParserBase.h"

class DefaultParser : public ParserBase<DefaultParser,0>
{
  friend class ParserBase<DefaultParser,0>;
protected:
  template<typename Functor, typename OtherArgs>
  bool parse(Functor& f, const OtherArgs& others) const
    {
    return this->defaultParse(f,others);
    };
};

#endif
