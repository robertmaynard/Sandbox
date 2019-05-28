Very small snippets of code to verify the vtkm::Vec generates vectorized
SIMD code.

To verify that we are getting vector SIMD code, build with the following flags:
- -DNDEBUG -O3 --std=c++11 -march=native -S