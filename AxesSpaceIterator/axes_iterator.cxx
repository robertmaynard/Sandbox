#include <algorithm>
#include <iostream>

#include "axes_iterator.h"

bool linear_iter::inc() { return false; }
void linear_iter::update_indices(
    std::vector<std::pair<axis, std::size_t>> &indices) const {}

bool tie_iter::inc() { return false; }
void tie_iter::update_indices(
    std::vector<std::pair<axis, std::size_t>> &indices) const {}

bool user_iter::inc() { return false; }
void user_iter::update_indices(
    std::vector<std::pair<axis, std::size_t>> &indices) const {}


void state_iterator::add_iteration_space(
    const axis_space_base &iteration_space) {}

void state_iterator::init() {}
bool state_iterator::iter_valid() const { return false; }

void state_iterator::next() {
  bool rolled_over = this->m_space[this->m_current_space].inc();
  if (rolled_over) {
    this->m_current_space += 1;
  }
  if (this->m_current_space == m_space.size()) {
    this->m_current_space = 0;
  }
}

std::vector<std::pair<axis, std::size_t>>
state_iterator::get_current_indices() const {
  std::vector<std::pair<axis, std::size_t>> indices(m_axes_count);
  for (auto &m : m_space) {
    m.update_indices(indices);
  }
  return indices;
}

int main() {
  axis x{"X", 4};
  axis y{"Y", 4};
  axis z{"Z", 4};

  // How to improve the iteration space API?
  state_iterator si;
  si.add_iteration_space(tie{{z, y}, {2, 1}});
  si.add_iteration_space(linear{x, 0});

  return 0;
}
