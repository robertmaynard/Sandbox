
#include <memory>

struct axis {
  std::string m_name;
  std::size_t m_size;
};

struct axis_space_base {};
struct linear : axis_space_base {
  axis m_axis;
  std::size_t m_index;
  linear(const axis &a, std::size_t index) : m_axis{a}, m_index{index} {}
};
struct tie : axis_space_base {
  std::vector<axis> m_axis;
  std::vector<std::size_t> m_indices;

  tie(const std::vector<axis> &a, std::vector<std::size_t> indices)
      : m_axis{a}, m_index{index} {}
};

struct user : axis_space_base {
  std::vector<axis> m_axis;
  std::vector<std::size_t> m_indices;

  user(const std::vector<axis> &a, std::vector<std::size_t> indices)
      : m_axis{a}, m_index{index} {}
};

struct axis_space_iterator {
  std::unique_ptr<axis_space_base> m_axes_space;
  std::size_t m_current_index;
  std::size_t m_size;


  virtual bool inc();
  virtual void
  update_indices(std::vector<std::pair<axis, std::size_t>> &indices) const;
};

struct linear_iter : axis_space_iterator {
  [[nodiscard]] bool inc() override;
  void update_indices(
      std::vector<std::pair<axis, std::size_t>> &indices) const override;
};
struct tie_iter : axis_space_iterator {
  [[nodiscard]] bool inc() override;
  void update_indices(
      std::vector<std::pair<axis, std::size_t>> &indices) const override;
};
struct user_iter : axis_space_iterator {
  [[nodiscard]] bool inc() override;
  void update_indices(
      std::vector<std::pair<axis, std::size_t>> &indices) const override;
};

struct state_iterator {
  void add_iteration_space(const   &iteration_space);

  void init();
  [[nodiscard]] bool iter_valid() const;
  void next();

  [[nodiscard]] std::vector<std::pair<axis, std::size_t>>
  get_current_indices() const;

  std::vector<axis_space_iterator> m_space;
  std::size_t m_axes_count{};
  std::size_t m_current_space{};
};
