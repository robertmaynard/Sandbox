
#include <type_traits>

namespace meta {
template <bool B> using bool_ = std::integral_constant<bool, B>;
}

template <class T> struct bool_ {
  using type = bool_;
  using value_type = bool;
  constexpr operator bool() const noexcept { return static_cast<bool>(T()); }
  constexpr bool operator()() const noexcept {
    return static_cast<bool>(*this);
  }
  template <bool B, typename T_ = T,
            typename std::enable_if<B == (bool)T_(), int>::type = 0>
  constexpr operator meta::bool_<B>() const noexcept {
    return {};
  }
};

// slow code
struct dummy {
  template <typename... Ts> constexpr dummy(Ts &&...) noexcept {}
  template <typename That,
            typename std::enable_if<(bool)That(), int>::type = 352>
  constexpr friend dummy operator&&(dummy, bool_<That>) noexcept {
    return {};
  }
};

int main(int, char **)
{
  dummy d;
  bool_<std::is_same<float,float>::type> b;
  auto t = d && b;
  return 0;
}
