project(ParallelForEach)

add_library(ParallelExecuteImpl ParallelExecute.cxx)

add_executable(ParallelForEach main.cxx)
target_link_libraries(ParallelForEach ParallelExecuteImpl)
