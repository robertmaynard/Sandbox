#include "Functor.h"
#include "Modify.h"
#include "ExtendFunctor.h"
#include "BuildSignature.h"

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

  typedef BuildSignature<typename ModifiedType::ControlSignature> NewContSig;
  typedef BuildSignature<typename ModifiedType::ExecutionSignature> NewExecSig;


  typedef ExtendedFunctor<Functor,NewContSig,NewExecSig> RealFunctor;

  typedef ConvertToBoost<RealFunctor> BoostExtendFunctor;

  std::cout << "new functor signatures" << std::endl;
  boost::mpl::for_each< typename BoostExtendFunctor::ControlSignature >(print_class_name());
  std::cout << std::endl;
  boost::mpl::for_each< typename BoostExtendFunctor::ExecutionSignature >(print_class_name());
  std::cout << std::endl;


  //also do a compile time verifification that the ExtendFunctor method work properly
  typedef VerifyTypes<typename ModifiedType::ExecutionSignature,
                      typename BoostExtendFunctor::ExecutionSignature> ExecSigVerified;
  typedef VerifyTypes<typename ModifiedType::ControlSignature,
                      typename BoostExtendFunctor::ControlSignature> ContSigVerified;
  }
};


int main()
{
  typedef verifyFunctor<functor::Derived> Verified;
  Verified v; v();

  std::cout  << std::endl << "VerifiedTwo" << std::endl;
  typedef verifyFunctor<functor::DerivedReturn> VerifiedTwo;
  VerifiedTwo vt; vt();

  std::cout  << std::endl << "VerifiedLotsOfArgs" << std::endl;
  typedef verifyFunctor<functor::DerivedLotsOfArgs> VerifiedLotsArgs;
  VerifiedLotsArgs va; va();
}
