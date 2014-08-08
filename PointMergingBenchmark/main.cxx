
#include "MergingTypes.h"
#include "PointGenerator.h"

#include <iostream>

template<typename T>
void test_generator(PointGenerator<T>& generator, Timer& timer)
{
  merging::VectorLowerBounds(generator);
  std::cout << "LB <<  " << timer.GetElapsedTime() << std::endl;

  timer.Reset();
  merging::VectorIntoDict(generator);
  std::cout << "Map <<  " << timer.GetElapsedTime() << std::endl;
  timer.Reset();
}

int main(int argc, char **argv)
{
  const float duplicate_point_ratio = 0.33f; //33% duplicate points
  const int number_of_points = 10000000; //lets start with ten million

  typedef PointGenerator<generator::Point1D> Generator1D;
  typedef PointGenerator<generator::Point3D> Generator3D;

  Timer timer;
  std::cout << "Testing out 1D generator" << std::endl;
  {
  Generator1D generator1d(duplicate_point_ratio,number_of_points);
  test_generator(generator1d, timer);
  }

  std::cout << "Testing out 3D generator" << std::endl;
  {
  Generator3D generator3d(duplicate_point_ratio,number_of_points);
  test_generator(generator3d, timer);

  merging::Locator(generator3d);
  std::cout << "Locator Octree <<  " << timer.GetElapsedTime() << std::endl;
  timer.Reset();


  merging::LocatorMP(generator3d);
  std::cout << "Locator MergePoints <<  " << timer.GetElapsedTime() << std::endl;
  }


  return 0;
}
