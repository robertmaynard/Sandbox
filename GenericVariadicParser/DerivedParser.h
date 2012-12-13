#ifndef __DerivedParser_h
#define __DerivedParser_h

#include "BaseParser.h"

#include <boost/mpl/assert.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/type_traits/is_same.hpp>


class DerivedParser : public BaseParser<DerivedParser,2>
{
  friend class BaseParser<DerivedParser,2>;
protected:
  template<typename Channel, typename Arg1, typename Arg2, typename OtherArgs>
  bool parse(Channel& c, const Arg1& one, const Arg2& two,
             const OtherArgs& others) const
    {
    return this->defaultParse(c,one,two,others);
    };

  template<typename Channel, typename OtherArgs>
  bool parse(Channel& c, const OtherArgs& others) const
    {
    return this->defaultParse(c,others);
    };

};

#endif
