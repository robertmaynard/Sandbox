#include <iostream>
#include <algorithm>

#include <boost/type_traits.hpp>

#include "Bindings.h"

int main()
{
  typedef void ControlSig(dax::cont::arg::Field(dax::cont::sig::In));
  typedef dax::internal::GetNthType<1, ControlSig>::type ParameterType;
  typedef dax::cont::internal::detail::GetConceptAndTagsImpl<ParameterType> ConceptAndTagsImpl;
  typedef ConceptAndTagsImpl::Tags Tags;
  // typedef dax::cont::arg::ConceptMap<ConceptAndTags,dax::cont::ArrayHandle<dax::Scalar> > type;
  // char b[sizeof(type)+1];
  return 0;
}

