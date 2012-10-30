#include "Functor.h"
#include "Modify.h"

 #include <boost/mpl/assert.hpp>
#include <boost/type_traits/is_same.hpp>

int main()
{
  typedef ReplaceAndExtend<functor::Derived,
                 arg::Replace,
                 arg::InsertedArg> ModifiedType;

  typedef GetTypes<ModifiedType::ExecutionSignature> ETypes;
  typedef GetTypes<ModifiedType::ControlSignature> CTypes;

  //confirm that we inserted _2 to execution signature
  BOOST_MPL_ASSERT(( boost::is_same<ETypes::Arg2Type,arg::Arg<2> > ));

  //confirm that we insterted the InsertedArg type to control signature
  BOOST_MPL_ASSERT(( boost::is_same<CTypes::Arg2Type,arg::InsertedArg > ));


  typedef ExtendFunctor<functor::Derived,ModifiedType> RealFunctor;

  typedef ConvertToBoost<RealFunctor> BoostExtendFunctor;
  typedef GetTypes<BoostExtendFunctor::ExecutionSignature> ExtendedETypes;
  typedef GetTypes<BoostExtendFunctor::ControlSignature> ExtendedCTypes;

  BOOST_MPL_ASSERT(( boost::is_same<ETypes::Arg2Type,ExtendedETypes::Arg2Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<CTypes::Arg2Type,ExtendedCTypes::Arg2Type > ));
}
