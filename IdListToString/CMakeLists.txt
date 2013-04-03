cmake_minimum_required(VERSION 2.8)
project(IdListToString)

#most likely overlinking here but I don't want to figure out
#the subset i require
find_package(VTK  COMPONENTS
  vtkCommonCore
  vtkCommonDataModel
  vtkCommonExecutionModel
  vtkCommonMisc
  REQUIRED)

include(${VTK_USE_FILE})

set(srcs
  convert.h
  main.cxx)

add_executable(IdListToStringTest ${srcs})
target_link_libraries(IdListToStringTest
  vtkCommonCore
  vtkCommonDataModel
  vtkCommonExecutionModel
  vtkCommonMisc)

