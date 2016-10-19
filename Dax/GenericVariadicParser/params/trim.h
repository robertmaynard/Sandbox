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
    enum {r_size =
      params::detail::num_elements<Sequence>::value - Leading_Number_To_Remove};
  public:
    typedef typename params::make_lview<Sequence,Leading_Number_To_Remove>::type LeadingView;

    typedef typename params::make_rview<Sequence,r_size>::type TrailingView;

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
