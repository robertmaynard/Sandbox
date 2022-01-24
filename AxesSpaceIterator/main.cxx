
#include "axes_iterator.h"
#include <iostream>

#include "axes_iterator.h"
#include <iostream>

void print(state_iterator &si) {
  // print header
  si.init();
  auto indices = si.get_current_indices();
  for (auto &&i : indices) {
    std::cout << i.first.m_name << "\t";
  }

  // print contents
  std::cout << "\n";
  for (; si.iter_valid(); si.next()) {
    indices = si.get_current_indices();
    for (auto &&i : indices) {
      std::cout << i.second << "\t";
    }
    std::cout << std::endl;
  }
}

int main() {
  axis x{"X", 4};
  axis y{"Y", 4};
  axis z{"Z", 4};

  {
    state_iterator si;
    si.add_iteration_space(tie{{z, y}, {2, 1}});
    si.add_iteration_space(linear{x, 0});

    print(si);
  }

  std::cout << std::endl << std::endl << std::endl;
  {
    state_iterator si;
    std::size_t y_pos = 0, z_pos = 0, y_start = 0;
    auto diag_under_adv = [&](std::size_t &current_index,
                              std::size_t max_length) -> bool {
      current_index++;
      y_pos++;
      if (y_pos == y.m_length) {
        y_pos = ++y_start;
        z_pos = y_start;
        return true;
      }
      return false;
    };
    auto diag_under = [&](std::size_t inc_index,
                          std::vector<std::pair<axis, std::size_t>> &indices) {
      indices[1] = {y, y_pos};
      indices[2] = {z, z_pos};
    };
    size_t iteration_length = ((y.m_length * (z.m_length + 1)) / 2);

    si.add_iteration_space(linear{x, 0});
    si.add_iteration_space(
        user{2, iteration_length, diag_under_adv, diag_under});

    print(si);
  }

  return 0;
}
