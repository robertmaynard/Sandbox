

#include <limits>

#ifdef MIX_VIS
struct __attribute__((visibility("default"))) RangeBase {
#else
struct  RangeBase {
#endif
  virtual ~RangeBase() {}

  virtual bool IsNonEmpty() const = 0;
};

template <typename T>
struct Range : public RangeBase {
  T Min;
  T Max;

  Range()
      : Min(),
        Max() {}

  ~Range() override {}

  template Range(const T &min, const T &max) : Min(min), Max(max) {}

  bool IsNonEmpty() const override { return (this->Min <= this->Max); }
};



