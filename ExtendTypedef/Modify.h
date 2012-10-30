
#ifndef __modify_
#define __modify_

#include "Arg.h"

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
  typedef boost::function_types::parameter_types<
            typename Functor::ControlSignature> ControlSignature;

  typedef boost::function_types::parameter_types<
            typename Functor::ExecutionSignature> ExecutionSignature;

  typedef boost::mpl::size<ControlSignature>   ContSize;
  typedef boost::mpl::size<ExecutionSignature> ExecSize;
};

template<int Value>
struct IntToArg
{
  typedef arg::Arg<Value> type;
};

template<typename Signature>
struct GetTypes
{
  typedef typename boost::mpl::at_c<Signature,0>::type Arg1Type;
  typedef typename boost::mpl::at_c<Signature,1>::type Arg2Type;
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

template<typename Functor, typename ExecArgToReplace>
struct Modify
{
  typedef ConvertToBoost<Functor> BoostTypes;
  typedef typename BoostTypes::ContSize ContSize;

  //walk the ExecutionControlSignature and find all the elements
  //that match the ExecArgToReplace
  typedef typename boost::mpl::plus<ContSize,
          boost::mpl::int_<1> >::type NewPlaceHolder;

  //We replace each element that matches the ExecArgToReplace types
  //with the ::arg::Arg<NewPlaceHolder> which we are going to next
  //push back into the control signature
  typedef typename boost::mpl::transform<
            typename BoostTypes::ExecutionSignature,
            detail::replace<ExecArgToReplace,
                            NewPlaceHolder::value >
            >::type ExecutionSignature;

  //now we have to extend the control signature to be one larger,
  //and to have the proper type added
  typedef typename boost::mpl::push_back<
            typename BoostTypes::ControlSignature,
            arg::InsertedArg>::type ControlSignature;

};





#endif
