#include "ParallelExecute.h"

void parallel_execute_impl(funcptr_t fp, const void *foptr, int start, int end)
{
# pragma omp paralel for
  for (int i = start; i < end; ++i)
    {
      fp(foptr, i);
    }
}

