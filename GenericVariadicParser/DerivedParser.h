#ifndef __DerivedParser_h
#define __DerivedParser_h

#include "ParserBase.h"
#include "Functor.h"

class DerivedParser : public ParserBase<DerivedParser,2>
{
  friend class ParserBase<DerivedParser,2>;
protected:
  template<typename Functor, typename Arg1, typename Arg2, typename OtherArgs>
  bool parse(Functor& f, const Arg1& one, Arg2& two,
             const OtherArgs& others) const
    {
    functor::NewFunctorType nf;
    Arg2 nt = two + two;
    return this->defaultParse(nf,one,nt,others);
    };

};

#endif
