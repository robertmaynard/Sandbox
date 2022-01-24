
#pragma once
#include <functional>
#include <memory>
#include <vector>

#include <iostream>

struct axis {
  std::string m_name;
  std::size_t m_length;
};

struct axis_space_iterator {
  using AdvanceSignature = bool(std::size_t &current_index, std::size_t length);
  using UpdateSignature = void(
      std::size_t index, std::vector<std::pair<axis, std::size_t>> &indices);

  [[nodiscard]] bool inc() {
    return this->m_advance(m_current_index, m_iteration_size);
  }

  void
  update_indices(std::vector<std::pair<axis, std::size_t>> &indices) const {
    this->m_update(m_current_index, indices);
  }

  std::size_t m_number_of_axes = 1;
  std::size_t m_iteration_size = 1;
  std::function<AdvanceSignature> m_advance = [](std::size_t &current_index,
                                                 std::size_t const &length) {
    (current_index + 1 == length) ? current_index = 0 : current_index++;
    return (current_index == 0); // we rolled over
  };
  std::function<UpdateSignature> m_update = nullptr;

private:
  std::size_t m_current_index = 0;
};

inline axis_space_iterator make_space_iterator(
    std::size_t axes_count, std::size_t iter_count,
    std::function<axis_space_iterator::AdvanceSignature> &&advance,
    std::function<axis_space_iterator::UpdateSignature> &&update) {

  axis_space_iterator iter;
  iter.m_number_of_axes = axes_count;
  iter.m_iteration_size = iter_count;
  iter.m_advance = std::move(advance);
  iter.m_update = std::move(update);
  return iter;
}

inline axis_space_iterator make_space_iterator(
    std::size_t axes_count, std::size_t iter_count,
    std::function<axis_space_iterator::UpdateSignature> &&update) {

  axis_space_iterator iter;
  iter.m_number_of_axes = axes_count;
  iter.m_iteration_size = iter_count;
  iter.m_update = std::move(update);
  return iter;
}

struct axis_space_base {
  using AdvanceSignature = axis_space_iterator::AdvanceSignature;
  using UpdateSignature = axis_space_iterator::UpdateSignature;
  axis_space_iterator m_iter;
};

struct linear : axis_space_base {
  linear(const axis &a, std::size_t index);
};
struct tie : axis_space_base {
  tie(const std::vector<axis> &a, std::vector<std::size_t> indices);
};
struct user : axis_space_base {
  user(std::size_t number_of_axes, std::size_t number_of_iterations,
       std::function<AdvanceSignature> &&advance_func,
       std::function<UpdateSignature> &&update_func);

  user(std::size_t number_of_axes, std::size_t number_of_iterations,
       std::function<UpdateSignature> &&update_func);
};

struct state_iterator {
  void add_iteration_space(const axis_space_base &iteration_space);

  void init();
  [[nodiscard]] bool iter_valid() const;
  void next();

  [[nodiscard]] std::vector<std::pair<axis, std::size_t>>
  get_current_indices() const;

  std::vector<axis_space_iterator> m_space;
  std::size_t m_axes_count = 0;
  std::size_t m_current_space = 0;
  std::size_t m_current_iteration = 0;
  std::size_t m_max_iteration = 1;
};
