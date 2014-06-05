cmake_minimum_required(VERSION 2.8.10)

project(ComputeRangeBenchmark CXX)

#dirty way to enable C++11, since we need that
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")

find_package(VTK
    COMPONENTS
    vtkCommonCore)

if(VTK_FOUND)
  include(${VTK_USE_FILE})
  add_executable(ComputeRangeBenchmark main.cxx)
  target_link_libraries(ComputeRangeBenchmark vtkCommonCore)

  if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_target_properties(ComputeRangeBenchmark
             PROPERTIES COMPILE_FLAGS "-std=c++11 -stdlib=libc++"
                        LINK_FLAGS    "-stdlib=libc++")
  else()
    set_target_properties(ComputeRangeBenchmark
             PROPERTIES COMPILE_FLAGS "-std=c++11")
  endif()
endif()



