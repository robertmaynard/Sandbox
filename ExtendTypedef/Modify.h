
#ifndef __modify_
#define __modify_

#include "Arg.h"

#include <boost/function_types/components.hpp>
#include <boost/function_types/parameter_types.hpp>

#include <boost/type_traits/is_same.hpp>

#include <boost/mpl/at.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector_c.hpp>

template<typename Functor>
struct ConvertToBoost
{
  typedef boost::function_types::components<
            typename Functor::ControlSignature> ControlSignature;

  typedef boost::function_types::components<
            typename Functor::ExecutionSignature> ExecutionSignature;

  typedef boost::mpl::size<ControlSignature>   ContSize;
  typedef boost::mpl::size<ExecutionSignature> ExecSize;
};

template<int Value>
struct IntToArg
{
  typedef arg::Arg<Value> type;
};

namespace detail
{
  template<typename ExecArgToReplace, int Value>
  struct replace
  {
  template<typename Arg>
  struct apply
    {
    typedef boost::is_same<ExecArgToReplace, Arg> AreSame;
    typedef typename IntToArg<Value>::type ReplacementArg;
    typedef typename boost::mpl::if_<AreSame,
                              ReplacementArg,
                              Arg>::type type;

    };
  };
}

template<typename Functor, typename ExecArgToReplace, typename ControlArgToUse>
struct ReplaceAndExtendSignatures
{
private:
  typedef ConvertToBoost<Functor> BoostTypes;
  typedef typename BoostTypes::ContSize NewPlaceHolderPos;

public:
  //We replace each element that matches the ExecArgToReplace types
  //with the ::arg::Arg<NewPlaceHolder> which we are going to next
  //push back into the control signature
  typedef typename boost::mpl::transform<
            typename BoostTypes::ExecutionSignature,
            detail::replace<ExecArgToReplace,
                            NewPlaceHolderPos::value >
            >::type ExecutionSignature;

  //now we have to extend the control signature to be one larger,
  //and to have the proper type added
  typedef typename boost::mpl::push_back<
            typename BoostTypes::ControlSignature,
            ControlArgToUse>::type ControlSignature;


  typedef boost::mpl::size<ControlSignature>   ContSize;
  typedef boost::mpl::size<ExecutionSignature> ExecSize;

};

template<typename Signature>
struct GetTypes
{
  typedef typename boost::mpl::at_c<Signature,0>::type Arg0Type;
  typedef typename boost::mpl::at_c<Signature,1>::type Arg1Type;
  typedef typename boost::mpl::at_c<Signature,2>::type Arg2Type;
  typedef typename boost::mpl::at_c<Signature,3>::type Arg3Type;
  typedef typename boost::mpl::at_c<Signature,4>::type Arg4Type;
  typedef typename boost::mpl::at_c<Signature,5>::type Arg5Type;
  typedef typename boost::mpl::at_c<Signature,6>::type Arg6Type;
  typedef typename boost::mpl::at_c<Signature,7>::type Arg7Type;
  typedef typename boost::mpl::at_c<Signature,8>::type Arg8Type;
  typedef typename boost::mpl::at_c<Signature,9>::type Arg9Type;
};

//next step is to convert the boost mpl types back to a worklet
//signature. To get this to work with all functor we need to use
//boost pre-processor
template<typename Functor, typename ExtendedFunctorSigs>
struct ExtendFunctor : public Functor
{
private:
  typedef typename ExtendedFunctorSigs::ControlSignature CSig;
  typedef typename ExtendedFunctorSigs::ExecutionSignature ESig;

public:
  typedef typename boost::mpl::at_c<CSig,0>::type
          ControlSignature(typename boost::mpl::at_c<CSig,1>::type);
  typedef typename boost::mpl::at_c<ESig,0>::type
          ExecutionSignature(typename boost::mpl::at_c<ESig,1>::type);

};





#endif
