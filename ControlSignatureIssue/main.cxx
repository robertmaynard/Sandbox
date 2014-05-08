#include <iostream>
#include <algorithm>

#include <boost/type_traits.hpp>

#include "Bindings.h"

namespace dax { namespace cont {
  namespace arg
    {
    class Field {};
    }
  namespace sig
    {
    class Tag {};
    class In: public Tag {};
    class Out: public Tag {};
    }
}}

int main()
{
  typedef void ControlSig(dax::cont::arg::Field(dax::cont::sig::In));
  typedef dax::internal::GetNthType<1, ControlSig>::type ParameterType;
  typedef dax::cont::internal::detail::GetConceptAndTagsImpl<ParameterType> ConceptAndTagsImpl;
  // typedef dax::cont::arg::ConceptMap<ConceptAndTags,dax::cont::ArrayHandle<dax::Scalar> > type;
  // char b[sizeof(type)+1];
  return 0;
}

