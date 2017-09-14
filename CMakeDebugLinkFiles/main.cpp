
#include "functions.h"

#include <iostream>
#include <iomanip>

//----------------------------------------------------------------------------
void test_range(double range[2] )
{
  std::cout << "------------------------------------------------------------" << std::endl;
  std::cout << "input range: " << range[0] << " : " << range[1] << std::endl;
  AdjustRange(range);
  std::cout << "a output range: " << std::fixed << std::setprecision(42) << range[0] << " : " << range[1] << std::endl;
  std::cout << std::endl;
}

int main(int, char **)
{
  double zeros[2] = { 0.0, 0.0 };
  double ones[2] = { 1.0, 1.0 };
  double nones[2] = { -1.0, -1.0 };
  double zero_one[2] = { 0.0, 1.0 };
  double none_one[2] = { -1.0, 1.0 };
  double none_zero[2] = { -1.0, 1.0 };

  double small[2] = { -12, -4};
  double large[2] = { 1e12, 1e12+1 };
  double large_exact[2] = { 1e12, 1e12 };
  double real_small[2] = { 1e-20, 1e-19 };

  test_range(zeros);
  test_range(ones);
  test_range(nones);
  test_range(zero_one);
  test_range(none_one);
  test_range(none_zero);

  test_range(small);
  test_range(large);
  test_range(large_exact);
  test_range(real_small);

  return 0;
}

