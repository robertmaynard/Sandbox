
#include "vtkm/Types.h"
#include <cmath>
#include <iostream>

//clang++ -DNDEBUG -O3 --std=c++11 -march=native -S mult.cpp

typedef vtkm::Vec<float, 3> Vec3;

int
main(int len, char**)
{
  if (len < 2)
  {
    len = 1048576;
  }

  Vec3* l = new Vec3[len];
  Vec3* o = new Vec3[len];
  float* st = new float[len];

  for (int i = 0; i < len; ++i)
  {
    l[i] = Vec3(i*2);
    o[i] = Vec3(i);
  }

  {
    asm("nop #opt 1");
    Vec3 osum(0);
    asm("nop #opt 2");
    for (int i = 0; i < len; ++i)
    {
      Vec3 t = o[i] * l[i];
      osum = osum + t;
    }
    std::cout << "sum: " << osum << "  --  " << st[3] << std::endl;
  }
}
