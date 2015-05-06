#ifndef ParallelExecute_h
#define ParallelExecute_h

typedef void (*funcptr_t)(const void*, int i);

void parallel_execute_impl(funcptr_t fp, const void *foptr, int start, int end);

#endif

