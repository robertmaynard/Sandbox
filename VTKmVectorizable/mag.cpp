
#include "vtkm/Types.h"
#include <cmath>
#include <iostream>

//Currently to packed vector for sqrt you have to enable fast math
//clang++ -DNDEBUG -O3 --std=c++11 -march=native -ffast-math -S mag.cpp

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
    float fsum=0;
    asm("nop #opt 2");
    for (int i = 0; i < len; ++i)
    {
      float t = vtkm::dot(o[i], l[i]);
      fsum += std::sqrt(t);


    }
    std::cout << "sum: " << fsum << "  --  " << st[3] << std::endl;
  }
}
