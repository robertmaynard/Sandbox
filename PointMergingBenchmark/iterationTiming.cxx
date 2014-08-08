

#include "PointGenerator.h"
#include "Timer.h"

#include <vector>
#include <set>
#include <list>
#include <iostream>
#include <algorithm>

int main(int argc, char **argv)
{
  const float duplicate_point_ratio = 0.0; //0% duplicate points
  const int number_of_points = 10000000; //lets start with ten million

  Timer timer;
  PointGenerator generator(duplicate_point_ratio,number_of_points);
  std::cout << "gen time <<  " << timer.GetElapsedTime() << std::endl;
  timer.Reset();

  std::set< generator::Point > pointSet(generator.begin(),generator.end());
  std::cout << "map gen time <<  " << timer.GetElapsedTime() << std::endl;

  timer.Reset();

  generator::Point min_p(number_of_points+1,number_of_points+1,number_of_points+1);
  for(int times=0; times < 10; ++times)
  {
  for(std::set< generator::Point >::const_iterator i = pointSet.begin();
      i != pointSet.end();
      i++) { min_p = std::min(*i,min_p); }
  }

  std::cout << "map iteration time:  " << timer.GetElapsedTime() << std::endl;

  timer.Reset();
  for(int times=0; times < 10; ++times)
  {
  for(PointGenerator::const_iterator i = generator.begin();
      i != generator.end();
      i++) { min_p = std::min(*i,min_p); }
  }
  std::cout << "vector iteration time:  " << timer.GetElapsedTime() << std::endl;

  std::list< generator::Point > pointList(generator.begin(), generator.end());
  timer.Reset();
  for(int times=0; times < 10; ++times)
  {
  for(std::list< generator::Point >::const_iterator i = pointList.begin();
      i != pointList.end();
      i++) { min_p = std::min(*i,min_p); }
  }

  std::cout << "list iteration time:  " << timer.GetElapsedTime() << std::endl;

  return 0;
}
