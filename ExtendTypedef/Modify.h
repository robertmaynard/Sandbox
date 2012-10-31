
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

template<typename Sig>
struct GetTypes
{
  typedef typename boost::mpl::at_c<Sig,0>::type Arg0Type;
  typedef typename boost::mpl::at_c<Sig,1>::type Arg1Type;
  typedef typename boost::mpl::at_c<Sig,2>::type Arg2Type;
  typedef typename boost::mpl::at_c<Sig,3>::type Arg3Type;
  typedef typename boost::mpl::at_c<Sig,4>::type Arg4Type;
  typedef typename boost::mpl::at_c<Sig,5>::type Arg5Type;
  typedef typename boost::mpl::at_c<Sig,6>::type Arg6Type;
  typedef typename boost::mpl::at_c<Sig,7>::type Arg7Type;
  typedef typename boost::mpl::at_c<Sig,8>::type Arg8Type;
  typedef typename boost::mpl::at_c<Sig,9>::type Arg9Type;
};
template<typename Sig1, typename Sig2>
struct VerifyTypes
{
  typedef GetTypes<Sig1> Sig1Types;
  typedef GetTypes<Sig2> Sig2Types;

  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg0Type, typename Sig2Types::Arg0Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg1Type, typename Sig2Types::Arg1Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg2Type, typename Sig2Types::Arg2Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg3Type, typename Sig2Types::Arg3Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg4Type, typename Sig2Types::Arg4Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg5Type, typename Sig2Types::Arg5Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg6Type, typename Sig2Types::Arg6Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg7Type, typename Sig2Types::Arg7Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg8Type, typename Sig2Types::Arg8Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg9Type, typename Sig2Types::Arg9Type > ));
};

#endif
