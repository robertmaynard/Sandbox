#include <algorithm>
#include <iostream>


struct axis
{
  std::string m_name;
  std::size_t m_size;
};

struct axis_space_base {};
struct linear : axis_space_base {};
struct diag : axis_space_base {};
struct tie : axis_space_base {};

struct axis_space_iterator
{
  axis_space_base m_axes_space;
  std::size_t m_current_index;
  std::size_t m_size;

  void inc();
  void update_indices(std::vector<std::size_t> indices)
};


struct state_iterator
{
  void add_axis(const axis &axis);

  [[nodiscard]] std::size_t get_number_of_states() const;

  void init();

  [[nodiscard]] const std::vector<std::size_t> &get_current_indices() const;

  [[nodiscard]] bool iter_valid() const;

  void next();

  std::vector<axis_space_iterator> m_space;
  std::vector<std::size_t> m_indices;

  std::size_t m_current_space{};
  std::size_t m_total{};
};

int main()
{
    axis x{"X", 4};
    axis y{"Y", 4};
    axis z{"Z", 4};

    state_iterator si;
    si.add_axis(x);
    si.add_axis(y);
    si.add_axis(z);

    // How to improve the iteration space API?
    si.add_iteration_space( linear{x} );
    si.add_iteration_space( tie{ {x, y} } );
    return 0;
}
