#include "Functor.h"
#include "Modify.h"

#include <boost/mpl/assert.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/type_traits/is_same.hpp>

#include <iostream>

struct print_class_name {
    template <typename T>
    void operator()( T t ) const {
       std::cout << typeid(t).name() << ", ";
    }

};


template<typename Functor>
struct verifyFunctor
{
  void operator()()
  {
  std::cout << "origin functor signatures" << std::endl;
  typedef ConvertToBoost<Functor> OrigFunctor;
  boost::mpl::for_each< typename OrigFunctor::ControlSignature >(print_class_name());
  std::cout << std::endl;
  boost::mpl::for_each< typename OrigFunctor::ExecutionSignature >(print_class_name());
  std::cout << std::endl;

  typedef ReplaceAndExtendSignatures<Functor,
                 arg::Replace,
                 arg::InsertedArg> ModifiedType;
  typedef ExtendFunctor<Functor,ModifiedType> RealFunctor;

  typedef ConvertToBoost<RealFunctor> BoostExtendFunctor;

  std::cout << "new functor signatures" << std::endl;
  boost::mpl::for_each< typename BoostExtendFunctor::ControlSignature >(print_class_name());
  std::cout << std::endl;
  boost::mpl::for_each< typename BoostExtendFunctor::ExecutionSignature >(print_class_name());
  std::cout << std::endl;


  //also do a compile time verifification that the ExtendFunctor method work properly
  // typedef GetTypes<typename ModifiedType::ExecutionSignature> ETypes;
  // typedef GetTypes<typename ModifiedType::ControlSignature> CTypes;
  // typedef GetTypes<typename BoostExtendFunctor::ExecutionSignature> ExtendedETypes;
  // typedef GetTypes<typename BoostExtendFunctor::ControlSignature> ExtendedCTypes;
  // BOOST_MPL_ASSERT(( boost::is_same<typename ETypes::Arg0Type, typename ExtendedETypes::Arg0Type > ));
  // BOOST_MPL_ASSERT(( boost::is_same<typename CTypes::Arg0Type, typename ExtendedCTypes::Arg0Type > ));
  // BOOST_MPL_ASSERT(( boost::is_same<typename ETypes::Arg1Type, typename ExtendedETypes::Arg1Type > ));
  // BOOST_MPL_ASSERT(( boost::is_same<typename CTypes::Arg1Type, typename ExtendedCTypes::Arg1Type > ));
  // BOOST_MPL_ASSERT(( boost::is_same<typename ETypes::Arg2Type, typename ExtendedETypes::Arg2Type > ));
  // BOOST_MPL_ASSERT(( boost::is_same<typename CTypes::Arg2Type, typename ExtendedCTypes::Arg2Type > ));
  }
};


int main()
{
  typedef verifyFunctor<functor::Derived> Verified;
  Verified v; v();

  std::cout  << std::endl << "VerifiedTwo" << std::endl;
  typedef verifyFunctor<functor::DerivedReturn> VerifiedTwo;
  VerifiedTwo vt; vt();
}
