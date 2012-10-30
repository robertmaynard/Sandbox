
#ifndef __modify_
#define __modify_

#include "Arg.h"

#include <boost/function_types/parameter_types.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/size.hpp>

#include <boost/mpl/iter_fold.hpp>
#include <boost/mpl/vector_c.hpp>

template<typename Functor>
class ConvertToBoost
{
public:
  typedef boost::function_types::parameter_types<
            typename Functor::ControlSignature> ControlSignature;

  typedef boost::function_types::parameter_types<
            typename Functor::ExecutionSignature> ExecutionSignature;

  typedef boost::mpl::size<ControlSignature>   ContSize;
  typedef boost::mpl::size<ExecutionSignature> ExecSize;
};

template<typename Signature>
struct GetTypes
{

  typedef typename boost::mpl::at_c<Signature,0> Arg1Type;
  typedef typename boost::mpl::at_c<Signature,1> Arg2Type;
};







namespace detail
{
  template<typename Arg>
  class replace
  {
  public:
    template<typename Output, typename Index>
    struct apply
    {
      typedef Output type;
    };

  };


}

template<typename Functor, typename ExecArgToReplace>
class Modify
{
public:
  typedef ConvertToBoost<Functor> BoostTypes;
  typedef typename BoostTypes::ContSize ContSize;

  //walk the ExecutionControlSignature and find all the elements
  //that match the ExecArgToReplace
  typedef boost::mpl::plus<ContSize,boost::mpl::int_<1> > NewPlaceHolder;

  typedef boost::mpl::vector_c<std::size_t,NewPlaceHolder::value >
            startingState;

  typedef typename boost::mpl::iter_fold<typename BoostTypes::ExecutionSignature,
                             startingState,
                             detail::replace<ExecArgToReplace>
                           >::type ExecutionReplaceResult;

  typedef ExecutionReplaceResult ExecutionSignature;
};





#endif
