cmake_minimum_required(VERSION 2.8)

 find_package(VTK 6.1
     COMPONENTS
     vtkCommonDataModel)

set(headers
  PointGenerator.h
  )

set(vtk_headers
  MergingTypes.h)

set(srcs
  main.cxx
  )

if(VTK_FOUND)
  include(${VTK_USE_FILE})
  add_executable(PointMerging ${headers} ${vtk_headers} ${srcs})
  target_link_libraries(PointMerging
    vtkCommonDataModel
    )
endif()

add_executable(IterationTiming ${headers} iterationTiming.cxx)

