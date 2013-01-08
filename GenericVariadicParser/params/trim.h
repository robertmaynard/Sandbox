#ifndef __params_trim_h
#define __params_trim_h


#include "common.h"
#include "views.h"


namespace params
{
  template<class Sequence, int Leading_Number_To_Remove>
  struct trim
  {
  private:
    enum {size=params::detail::num_elements<Sequence>::value};
  public:
    typedef typename params::detail::make_nview<Sequence,0,Leading_Number_To_Remove>::type LeadingView;
    typedef typename params::detail::make_nview<Sequence,Leading_Number_To_Remove,size>::type TrailingView;

    LeadingView FrontArgs(Sequence& s) const
    {
    return LeadingView(s);
    }

    TrailingView BackArgs(Sequence& s) const
    {
    return TrailingView(s);
    }
  };
}

#endif
