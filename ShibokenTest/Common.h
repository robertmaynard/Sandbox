
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

  using std::tr1::shared_ptr;

  typedef a::shared_ptr< a::b::ShibokenItem > SharedItem;
}