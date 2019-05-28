
#include "vtkm/Types.h"
#include "vtkm/VectorAnalysis.h"
#include <cmath>
#include <iostream>

//clang++ -DNDEBUG -O3 --std=c++11 -march=native -S cross.cpp

typedef vtkm::Vec<float, 3> Vec3;

int
main(int len, char**)
{
  if (len < 2)
  {
    len = 1048576;
  }

  float* l = new float[len+1];
  float* st = new float[len];
  for (int i = 0; i < len+1; ++i)
  {
    l[i] = static_cast<float>(i*2);
  }

  Vec3* o = new Vec3[len];
  for (int i = 0; i < len; ++i)
  {
    o[i] = Vec3(i);
  }

  {
    asm("nop #opt 1");
    Vec3 osum(0);
    asm("nop #opt 2");
    for (int i = 0; i < len-1; ++i)
    {
      o[i] = vtkm::Cross(o[i+1], o[i]) * l[i+1];
      osum = osum + o[i];
    }
    std::cout << "sum: " << osum << "  --  " << st[3] << std::endl;
  }
}
