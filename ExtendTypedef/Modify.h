
#ifndef __modify_
#define __modify_

#include <boost/function_types/parameter_types.hpp>
#include <boost/mpl/at.hpp>

template<typename Functor>
class Modify
{
public:
  typedef boost::function_types::parameter_types<
            typename Functor::ControlSignature> PType;

  typedef typename boost::mpl::at_c<PType,0>::type Arg1Type;
  typedef typename boost::mpl::at_c<PType,1>::type Arg2Type;


};

#endif
