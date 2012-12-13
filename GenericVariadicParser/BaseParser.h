#ifndef __BaseParser_h
#define __BaseParser_h

#include <tr1/tuple>
#include <tr1/utility>
#include <utility>
#include <iostream>

namespace detail
{
template <class T1, class ...T>
struct first
{
    typedef T1 type;
};

template<class T1, class ...T>
struct last
{
  typedef typename last<T...>::type type;

  type operator()(T1, T... t) const
  {
    return last<T...>()(t...);
  }

};

template<class T1>
struct last<T1>
{
  typedef T1 type;

  type operator()(T1 t1) const
  {
    return t1;
  }

};


}

template<class Derived,int Seperate_Args>
class BaseParser
{
public:
  template<typename Channel, typename... Args>
  bool operator()(Channel& c, Args... args) const
  {
  //the basic operation is to strip N args
  //from the start of the variadic list and pass
  //those in a unique items to Derived class, and than
  //pack the rest in a tuple class

  typedef std::tr1::tuple<Args...> ArgTupleType;
  ArgTupleType tuple(args...);

  //forward the arguments to decrease copies
  return static_cast<const Derived*>(this)->parse(c,tuple);
  }
protected:
  template<typename Channel, typename... Args>
  bool defaultParse(Channel& c,Args... args) const
  {
    //subtle note, the items in args can be individual items, the unique
    //item is the list item which can be a varaidic tuple that needs to be
    //unpacked

    //construct an std::tuple from args.
    //extract the last element, and see if it a tuple itself
    //if tuple join its elements into the args elements ( recreate args? )

    //walk args and print each element by using a foreach call?
    typedef typename detail::last<Args...> FetchLastArg;
    typedef typename FetchLastArg::type LastArgType;
    LastArgType lastArg = FetchLastArg()(args...);

    return true;
  }


};

#endif
