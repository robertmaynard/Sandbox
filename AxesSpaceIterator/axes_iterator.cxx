
#include "axes_iterator.h"

linear::linear(const axis &a, std::size_t meta_index) {
  auto update_func = [=](std::size_t inc_index,
                         std::vector<std::pair<axis, std::size_t>> &indices) {
    indices[meta_index] = {a, inc_index};
  };

  this->m_iter = make_space_iterator(1, a.m_length, update_func);
}
tie::tie(const std::vector<axis> &axes, std::vector<std::size_t> meta_indices) {
  auto update_func = [=](std::size_t inc_index,
                         std::vector<std::pair<axis, std::size_t>> &indices) {
    for (std::size_t i = 0; i < axes.size(); ++i) {
      indices[meta_indices[i]] = {axes[i], inc_index};
    }
  };

  this->m_iter =
      make_space_iterator(axes.size(), axes[0].m_length, update_func);
}

user::user(std::size_t number_of_axes, std::size_t number_of_iterations,
           std::function<AdvanceSignature> &&adv_func,
           std::function<UpdateSignature> &&update_func) {
  this->m_iter = make_space_iterator(number_of_axes, number_of_iterations,
                                     std::move(adv_func),
                                     std::move(update_func));
}

user::user(std::size_t number_of_axes, std::size_t number_of_iterations,
           std::function<UpdateSignature> &&update_func) {
  this->m_iter = make_space_iterator(number_of_axes, number_of_iterations,
                                     std::move(update_func));
}

void state_iterator::add_iteration_space(
    const axis_space_base &iteration_space) {
  auto &iter = iteration_space.m_iter;
  this->m_space.push_back(iter);
  this->m_axes_count += iter.m_number_of_axes;
  this->m_max_iteration *= iter.m_iteration_size;
}

void state_iterator::init() {
  this->m_current_space = 0;
  this->m_current_iteration = 0;
}
bool state_iterator::iter_valid() const {
  return m_current_iteration < m_max_iteration;
}
void state_iterator::next() {
  m_current_iteration++;

  for (auto &&space : this->m_space) {
    auto rolled_over = space.inc();
    if (rolled_over) {
      continue;
    }
    break;
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
