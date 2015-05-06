#ifndef ParallelForEach_h
#define ParallelForEach_h

#include "ParallelExecute.h"
#include <iostream>

template <typename T>
void run_functor(const void *foptr, int i)
{
  const T &fo = *reinterpret_cast<const T*>(foptr);
  fo(i);
}


template <typename T>
void parallel_for_each(const T &fo, int start, int end)
{
  funcptr_t fptr = run_functor<T>;
  parallel_execute_impl(fptr, &fo, start, end);
}

#endif

