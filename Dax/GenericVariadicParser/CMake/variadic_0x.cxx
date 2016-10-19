#include <tr1/tuple>
#include <tr1/utility>

namespace
{

struct test_functor {
  template<class T>
  void operator()(T& t) const { t = T(); }
}; 

//holds a sequence of integers.
template<int ...>
struct sequence { };

//generate a sequence of incrementing values starting at zero
template<int N, int ...S>
struct generate_sequence
{
  typedef typename generate_sequence<N-1, N-1, S...>::type type;
};

//generate a sequence of incrementing values starting at zero
template<int ...S>
struct generate_sequence<0, S...>
{
  typedef sequence<S...> type;
};

//pre GCC 4.7 has a bug with template expansion into
//non-variadic class template (aka base case).
//see gcc bug 35722, for the workaround I am using.
template< template <class ...> class T, class... Args>
struct Join { typedef T<Args...> type; };


//apply a functor to each element in a parameter pack
template<class First, class ...T>
struct forEach
{
  template<typename Functor>
  void operator()(Functor f, First first, T... t) const
  {
    //pre GCC 4.7 has a bug with template expansion into 
    //non-variadic class template (aka base case).
    //see gcc bug 35722, for the workaround I am using.
    typedef typename ::Join< ::forEach,T...>::type ForEachType;
    f(first);
    ForEachType()(f,t...);
  }
};

//apply a functor to each element in a parameter pack
template<class First>
struct forEach<First>
{
  template<typename Functor>
  void operator()(Functor f, First first) const
  {
    f(first);
  }
};

//applies the functor to each element in a parameter pack
template<class  Functor, class ...T>
void for_each(Functor f, T... items)
{
  //pre GCC 4.7 has a bug with template expansion into
  //non-variadic class template (aka base case).
  //see gcc bug 35722, for the workaround I am using.
  typedef typename ::Join<forEach,T...>::type ForEachType;
  ForEachType fe;
  fe(f,items...);
}

//special version of for_each that is a helper to get the length of indicies
//as a parameter type. I can't figure out how to do this step inside for_each specialized
//on tuple
template<class Functor, class ...T, int ...Indices>
void for_each(Functor f, std::tr1::tuple<T...> tuple, ::sequence<Indices...>)
{
  ::for_each(f,std::tr1::get<Indices>(tuple)...);
}

//function overload that detects tuples being sent to for each
//and expands the tuple elements
template<class Functor, class ...T>
void for_each(Functor f, std::tr1::tuple<T...>& tuple)
{
  //to iterate each item in the tuple we have to convert back from
  //a tuple to a parameter pack
  enum { len = std::tr1::tuple_size< std::tr1::tuple<T...> >::value};
  typedef typename ::generate_sequence<len>::type SequenceType;
  ::for_each(f,tuple,SequenceType());
}


template<typename ...Values>
void InvokeTuple(Values... v)
{
  std::tr1::tuple<Values...> tuple(v...);
  ::for_each(::test_functor(),tuple);
}

}

int main(int argc, char** argv)
{
  InvokeTuple("0",'1',2.0f,3);
  return 0;
}
