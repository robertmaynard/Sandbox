
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


template<typename BInt>
struct BoostIntToArg
{
  typedef arg::Arg<BInt::value> type;
};

template<typename Signature>
struct GetTypes
{
  typedef typename boost::mpl::at_c<Signature,0>::type Arg1Type;
  typedef typename boost::mpl::at_c<Signature,1>::type Arg2Type;
};



namespace detail
{
  template<typename ExecArgToReplace, typename Value>
  struct replace
  {
  template<typename Arg>
  struct apply
    {
    typedef boost::is_same<ExecArgToReplace, Arg> AreSame;
    typedef typename BoostIntToArg<Value>::type ReplacementArg;
    typedef typename boost::mpl::if_<AreSame,
                              ReplacementArg,
                              Arg>::type type;
    //typedef Arg type;
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

  // typedef boost::mpl::vector_c<std::size_t,NewPlaceHolder::value >
  //           startingState;

  typedef typename boost::mpl::transform<
            typename BoostTypes::ExecutionSignature,
            detail::replace<ExecArgToReplace,NewPlaceHolder > >::type ExecutionReplaceResult;

  typedef ExecutionReplaceResult ExecutionSignature;
};





#endif
