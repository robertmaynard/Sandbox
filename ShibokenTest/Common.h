
#include <algorithm>
#include <set>
#include <map>
#include <tr1/memory>

namespace a
{
  namespace b
    {
    class ShibokenItem;
    }

  typedef std::set<int> IntSet;
  typedef std::tr1::shared_ptr< a::b::ShibokenItem > SharedItem;
}