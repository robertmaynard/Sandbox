#ifndef __DerivedParser_h
#define __DerivedParser_h

#include "ParserBase.h"
#include "Functor.h"

class DerivedParser : public ParserBase<DerivedParser,2>
{
  friend class ParserBase<DerivedParser,2>;
protected:
  template<typename Functor, typename Arg1, typename Arg2, typename OtherArgs>
  bool parse(Functor& f, const Arg1& one, const Arg2& two,
             const OtherArgs& others) const
    {
    functor::NewFunctorType nf;
    std::tr1::tuple<Arg1,Arg2> newArgs(one,two);
    return this->defaultParse(nf,newArgs,others,newArgs);
    };

};

#endif
