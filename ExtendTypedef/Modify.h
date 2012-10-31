
#ifndef __modify_
#define __modify_

#include "Arg.h"

#include <boost/function_types/components.hpp>
#include <boost/function_types/parameter_types.hpp>

#include <boost/type_traits/is_same.hpp>

#include <boost/mpl/at.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/replace.hpp>
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

namespace detail
{

  template<typename Value>
  struct MPLIntToArg
  {
    typedef arg::Arg<Value::value> type;
  };

  template<typename Sequence, typename OldType, typename NewType>
  struct Replace
  {
    //determine if the execution arg we are searching to replace exists
    typedef typename boost::mpl::contains<Sequence,OldType>::type found;

    //We replace each element that matches the ExecArgToReplace types
    //with the ::arg::Arg<NewPlaceHolder> which we are going to next
    //push back into the control signature
    typedef typename boost::mpl::replace<
              Sequence,
              OldType,
              NewType>::type type;
  };


  template<typename Sequence, typename Type>
  struct PushBack
  {
    //push back type to the given sequence
    typedef typename boost::mpl::push_back<Sequence,Type>::type type;
  };

}

template<typename Functor, typename ExecArgToReplace, typename ControlArgToUse>
struct ReplaceAndExtendSignatures
{
private:
  typedef ConvertToBoost<Functor> BoostTypes;
  typedef typename BoostTypes::ContSize NewPlaceHolderPos;
  typedef typename detail::MPLIntToArg<NewPlaceHolderPos>::type ReplacementArg;

  //create the struct that will return us the new control signature if
  //we find the exec arg in the exec signature. This is extracted
  //from the mpl::if_ to make it more readable
  typedef typename ::detail::PushBack<
        typename BoostTypes::ControlSignature, ControlArgToUse> PushBackContSig;
public:

  typedef ::detail::Replace<typename BoostTypes::ExecutionSignature,
                            ExecArgToReplace,
                            ReplacementArg> ReplacedExecSigArg;


  //expose our new execution signature
  typedef typename ReplacedExecSigArg::type ExecutionSignature;


  //check ReplacedExecSigArg to see if we did actually find the execArg
  //in the signature. If not found use the original control signature
  typedef typename boost::mpl::if_<
          typename ReplacedExecSigArg::found,
          typename PushBackContSig::type,
          typename BoostTypes::ControlSignature>::type ControlSignature;
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
