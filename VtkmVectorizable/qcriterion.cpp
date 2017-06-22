
#include "vtkm/Types.h"
#include <cmath>
#include <iostream>

// Currently to packed vector for sqrt you have to enable fast math
// clang++ -DNDEBUG -O3 --std=c++11 -march=native -ffast-math -S qcriterion.cpp

typedef vtkm::Vec<float, 3> Vec3;

int main(int len, char **) {
  if (len < 2) {
    len = 1048576;
  }

  Vec3 *x = new Vec3[len];
  Vec3 *y = new Vec3[len];
  Vec3 *z = new Vec3[len];
  float *qc = new float[len];

  for (int i = 0; i < len; ++i) {
    x[i] = Vec3(i * 2);
    y[i] = Vec3(i);
    z[i] = Vec3((i - 1) * 3);
  }

  {
    float fsum = 0;
    asm("nop");
    asm("nop");
    for (int i = 0; i < len; ++i) {
      const Vec3 v(z[i][1] - y[i][2], x[i][2] - z[i][0], y[i][0] - x[i][1]);
      const Vec3 s(z[i][1] + y[i][2], x[i][2] + z[i][0], y[i][0] + x[i][1]);
      const Vec3 d(x[i][0], y[i][1], z[i][2]);

      // compute QCriterion
      qc[i] = ((vtkm::dot(v, v) / 2.0f) -
               ((vtkm::dot(s, s) + vtkm::dot(d, d)) / 2.0f)) /
              2.0f;
      fsum += qc[i];
    }
    std::cout << "sum: " << fsum << "  --  " << qc[3] << std::endl;

    asm("nop");
    asm("nop");
    asm("nop");
    for (int i = 0; i < len; ++i) {
      float t1 = ((z[i][1] - y[i][2]) * (z[i][1] - y[i][2]) +
                  (y[i][0] - x[i][1]) * (y[i][0] - x[i][1]) +
                  (x[i][2] - z[i][0]) * (x[i][2] - z[i][0])) /
                 2.0f;
      float t2 = x[i][0] * x[i][0] + y[i][1] * y[i][1] + z[i][2] * z[i][2] +
                 ((z[i][1] + y[i][2]) * (z[i][1] + y[i][2]) +
                  (y[i][0] + x[i][1]) * (y[i][0] + x[i][1]) +
                  (x[i][2] + z[i][0]) * (x[i][2] + z[i][0])) /
                     2.0f;

      qc[i] = (t1 - t2) / 2.0f;
      fsum += qc[i];
    }
    std::cout << "sum: " << fsum << "  --  " << qc[3] << std::endl;
  }
}
