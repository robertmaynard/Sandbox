

#ifndef __extendFunctor_
#define __extendFunctor_

template<class Functor, typename CSig, typename ESig>
struct ExtendedFunctor : public Functor
{
public:
  typedef typename CSig::type ControlSignature;
  typedef typename ESig::type ExecutionSignature;
};

#endif
