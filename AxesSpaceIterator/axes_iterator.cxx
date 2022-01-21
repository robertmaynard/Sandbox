#include <algorithm>
#include <iostream>

#include "axes_iterator.h"

linear::linear(const axis &a, std::size_t meta_index)
{
  auto update_func = [=](std::size_t inc_index,
                        std::vector<std::pair<axis, std::size_t>> &indices) {

    indices[meta_index] = {a, inc_index};
  };

  this->m_iter = make_space_iterator(1, a.m_length, update_func);
}
tie::tie(const std::vector<axis> &axes, std::vector<std::size_t> meta_indices)
{
  auto update_func = [=](std::size_t inc_index,
                        std::vector<std::pair<axis, std::size_t>> &indices) {

    for(std::size_t i = 0; i < axes.size(); ++i)
    {
      indices[meta_indices[i]] = {axes[i], inc_index};
    }
  };

  this->m_iter = make_space_iterator(axes.size(), axes[0].m_length, update_func);
}
user::user(std::size_t number_of_axes,
           std::size_t number_of_iterations,
           std::function<UpdateSignature>&& update_func)
{
  this->m_iter = make_space_iterator(number_of_axes, number_of_iterations, std::move(update_func));
}

void state_iterator::add_iteration_space(
    const axis_space_base &iteration_space) {
  auto& iter = iteration_space.m_iter;
  this->m_space.push_back(iter);
  this->m_axes_count += iter.m_number_of_axes;
  this->m_max_iteration *= iter.m_iteration_size;
}

void state_iterator::init() {
  this->m_current_space = 0;
  this->m_current_iteration = 0;
  }
bool state_iterator::iter_valid() const { return m_current_iteration < m_max_iteration; }
void state_iterator::next() {
  m_current_iteration++;

  for(auto&& space : this->m_space)
  {
    auto rolled_over = space.inc();
    if(rolled_over) {
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

int main() {
  axis x{"X", 4};
  axis y{"Y", 4};
  axis z{"Z", 4};

  {
    state_iterator si;
    si.add_iteration_space(tie{{z, y}, {2, 1}});
    si.add_iteration_space(linear{x, 0});
    std::cout << "X\tY\tZ tie(y,z)" << std::endl;
    for(si.init(); si.iter_valid(); si.next())
    {
      auto indices = si.get_current_indices();
      for(auto&& i : indices) {
        std::cout << i.second << "\t";
      }
      std::cout << std::endl;
    }
  }

  std::cout << std::endl << std::endl << std::endl;
  {
    state_iterator si;
    auto diag_under = [&](std::size_t inc_index,
                          std::vector<std::pair<axis, std::size_t>> &indices){

      // I think we need a `inc()` lambda to make this easier to write
      std::size_t y_pos = 0, inc = 0;
      for(std::size_t i=0; i < inc_index; i++)
      {
        y_pos++;
        if(y_pos == y.m_length) {
          y_pos = ++inc;
        }
      }

      size_t z_pos = 0; inc = y.m_length + 1;
      for(std::size_t i=0; i <= inc_index; i+=inc)
      {
        z_pos = ((y.m_length + 1) - (inc));
        --inc;
      }
      indices[1] = {y, y_pos};
      indices[2] = {z, z_pos};

    };
    size_t iteration_length = ((y.m_length * (z.m_length+1))/2);

    std::cout << iteration_length << std::endl;
    si.add_iteration_space(linear{x, 0});
    si.add_iteration_space(user{2, iteration_length, diag_under });

    std::cout << "X\tY\tZ diag(y/z)" << std::endl;
    for(si.init(); si.iter_valid(); si.next())
    {
      auto indices = si.get_current_indices();
      for(auto&& i : indices) {
        std::cout << i.second << "\t";
      }
      std::cout << std::endl;
    }
  }

  return 0;
}
