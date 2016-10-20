
#include "vtkm/Types.h"
#include "vtkm/Math.h"
#include <cmath>
#include <iostream>

//clang++ -DNDEBUG -O3 --std=c++11 -march=native -S mult.cpp
//Add -ffast-math -mrecip to get it to use vrcpss/vrcpps
typedef vtkm::Vec<float, 2> Vec2;

static const float AngleList[5] =
  { 0.643501108793284,  // angle for 3, 4, 5 triangle.
    0.78539816339745,   // pi/4
    0.5235987755983,    // pi/6
    1.0471975511966,    // pi/3
    0.0 };
static const float AdjacentList[5] =   { 4.0, 1.0, 1.732050807568877 /*sqrt(3)*/, 1.0, 1.0 };
static const float HypotenuseList[5] = { 5.0, 1.414213562373095 /*sqrt(2)*/, 2.0, 2.0, 1.0 };

int main(int len, char**)
{

  for(std::size_t i=0; i < 4; ++i)
  {
    Vec2 angle(AngleList[i],AngleList[i+1]);
    Vec2 adjacent(AdjacentList[i],AdjacentList[i+1]);
    Vec2 hypotenuse(HypotenuseList[i],HypotenuseList[i+1]);

    Vec2 result = vtkm::ACos(adjacent/hypotenuse);
    std::cout << "expected angle: " << angle << std::endl;
    std::cout << "computed angle: " << result << std::endl;

    asm("nop #opt 1");
    Vec2 result2( std::acos((1/hypotenuse[0])*adjacent[0]),
                  std::acos((1/hypotenuse[1])*adjacent[1]) );
    std::cout << "computed angle2: " << result2 << std::endl;
  }

  return 0;
}
