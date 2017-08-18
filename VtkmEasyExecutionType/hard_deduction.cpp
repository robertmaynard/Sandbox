
// clang++ -O3 -DNDEBUG --std=c++11 hard_deduction.cpp

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

  template <typename Tag> struct ExecutionTypes {
    using Portal =
        typename ExecutionManagerType::template ExecutionTypes<Tag>::Portal;
    using PortalConst = typename ExecutionManagerType::template ExecutionTypes<
        Tag>::PortalConst;
  };

  template <typename Tag>
  typename ExecutionTypes<Tag>::Portal PrepareForInput(Tag) const {
    return typename ExecutionTypes<Tag>::Portal();
  }

  template <typename Tag>
  typename ExecutionTypes<Tag>::PortalConst PrepareForOutput(Tag) const {
    return typename ExecutionTypes<Tag>::Portal();
  }
};

template <typename T, typename Tag> struct UsePortal {
  using PortalType = typename HandleType::template ExecutionTypes<Tag>::Portal;

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
