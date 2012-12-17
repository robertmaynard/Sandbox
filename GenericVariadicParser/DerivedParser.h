#ifndef __DerivedParser_h
#define __DerivedParser_h

#include "BaseParser.h"

#include "../ExtendTypdef/Modify.h"
#include "../ExtendTypdef/ExtendFunctor.h"
#include "../ExtendTypdef/BuildSignature.h"


class DerivedParser : public BaseParser<DerivedParser,2>
{
  friend class BaseParser<DerivedParser,2>;
protected:
  template<typename Functor, typename Arg1, typename Arg2, typename OtherArgs>
  bool parse(Functor& f, const Arg1& one, const Arg2& two,
             const OtherArgs& others) const
    {
    typedef ReplaceAndExtendSignatures<Functor, arg::Replace, arg::InsertedArg> ModifiedType;

  typedef BuildSignature<typename ModifiedType::ControlSignature> NewContSig;
  typedef BuildSignature<typename ModifiedType::ExecutionSignature> NewExecSig;


  typedef ExtendedFunctor<Functor,NewContSig,NewExecSig> RealFunctor;

    std::tr1::tuple<Arg1,Arg2> newArgs(one,two);
    return this->defaultParse(new_Functor,newArgs,others,newArgs);
    };

};

#endif
