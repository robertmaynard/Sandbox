
// clang++ -O3 -DNDEBUG --std=c++11 easy_deduction.cpp

#include <vector>

template <typename T> struct ExecutionManager {
  template <typename DeviceAdapter> struct ExecutionTypes {
  public:
    using Portal = typename std::vector<T>::iterator;
    using PortalConst = typename std::vector<T>::const_iterator;
  };
};

struct ImplicitPortalType {
  using ValueType = void *;
  using IteratorType = void *;
};
template <> struct ExecutionManager<double> {
  template <typename DeviceAdapter> struct ExecutionTypes {
  public:
    using Portal = ImplicitPortalType;
    using PortalConst = typename std::vector<double>::const_iterator;
  };
};

template <typename T> struct HandleType {
  using ValueType = T;
  using ExecutionManagerType = ExecutionManager<T>;

  template <typename Tag>
  using Portal =
      typename ExecutionManagerType::template ExecutionTypes<Tag>::Portal;

  template <typename Tag>
  using PortalConst =
      typename ExecutionManagerType::template ExecutionTypes<Tag>::PortalConst;

  template <typename Tag> Portal<Tag> PrepareForInput(Tag) const {
    return Portal<Tag>();
  }

  template <typename Tag> PortalConst<Tag> PrepareForOutput(Tag) const {
    return PortalConst<Tag>();
  }
};

template <typename HType, typename Tag>
using PortalType = typename HType::template Portal<Tag>;

template <typename HType, typename Tag>
using PortalConstType = typename HType::template PortalConst<Tag>;

template <typename T, typename Tag> struct UsePortal {
  // version 1 use decltype(declval) still really really long
  // using PortalType =
  //     decltype(std::declval<HandleType<T>>().PrepareForInput(Tag()));

  // version 2 uses Portal<> from the handle type, and is long
  // using PortalType = typename HandleType<T>::template Portal<Tag>;

  // version 3 is better
  using PortalType = PortalType<HandleType<T>, Tag>;

  UsePortal(PortalType portal) : P(portal) {}

  PortalType P;
};

struct Tag1 {};

int main(int, char *[]) {

  HandleType<float> foo;
  auto input = foo.PrepareForInput(Tag1());
  auto output = foo.PrepareForOutput(Tag1());

  // 1. We need to verify the types
  static_assert(
      std::is_same<decltype(input), std::vector<float>::iterator>::value,
      "input type wrong");
  static_assert(
      std::is_same<decltype(output), std::vector<float>::const_iterator>::value,
      "output type wrong");

  // 2. We need to try and use a class that wants to hold onto the portal types
  // as a member variable
  HandleType<double> dfoo;
  auto dinput = dfoo.PrepareForInput(Tag1());
  UsePortal<double, Tag1> up(dinput);

  return 0;
}
