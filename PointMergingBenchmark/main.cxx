
#include "MergingTypes.h"
#include "PointGenerator.h"

#include <iostream>

int main(int argc, char **argv)
{
  const float duplicate_point_ratio = 0.20f; //66% duplicate points
  const int number_of_points = 10000000; //lets start with ten million

  Timer timer;
  PointGenerator generator(duplicate_point_ratio,number_of_points);

  timer.Reset();
  merging::VectorLowerBounds(generator);
  std::cout << "LB <<  " << timer.GetElapsedTime() << std::endl;

  timer.Reset();
  merging::Locator(generator);
  std::cout << "Locator Octree <<  " << timer.GetElapsedTime() << std::endl;


  timer.Reset();
  merging::LocatorMP(generator);
  std::cout << "Locator MergePoints <<  " << timer.GetElapsedTime() << std::endl;


  return 0;
}
