#ifndef __params_indices_h
#define __params_indices_h


///////////////////////////////////////////////////////////////////////////////
// Variadic Implementation
///////////////////////////////////////////////////////////////////////////////
#if defined(VARIADIC_SUPPORT)

namespace params
{
  //holds a sequence of integers.
  template<int ...>
  struct static_indices { };

  //generate a struct with template args of incrementing values from Start to End
  //which is inclusive at the start, exclusive at end.
  //start must be less than end
  template<int Start, int End, int ...S>
  struct make_indices { typedef typename make_indices<Start,End-1, End-1, S...>::type type; };

  //generate a struct with tempalte args of incrementing values from Start to End
  //which is inclusive at the start, exclusive at end.
  //start must be less than end
  template<int Start, int ...S>
  struct make_indices<Start,Start, S...> { typedef static_indices<S...> type; };
}

#endif // VARIADIC_SUPPORT

#endif // __params_indices_h
