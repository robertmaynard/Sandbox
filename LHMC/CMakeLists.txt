##=============================================================================
##
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2012 Sandia Corporation.
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##
##=============================================================================
cmake_minimum_required(VERSION 2.8.10)

project(LHMC CXX)

find_package(OpenGL)
find_package(GLEW)
find_package(GLUT)

find_package(Dax REQUIRED)
DaxConfigureSerial(REQUIRED)
DaxConfigureTBB()
DaxConfigureCuda( REQUIRED)

#-----------------------------------------------------------------------------
set(headers
  ArgumentsParser.h
  CoolWarmColorMap.h
  Mandlebulb.h
  ShaderProgram.h
  ShaderCode.h
  Window.h
  Worklets.h
  )

set(sources
  ArgumentsParser.cxx
  Mandlebulb.cxx
  Window.cxx
  main.cxx
  )

#-----------------------------------------------------------------------------
set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY TRUE)

#-----------------------------------------------------------------------------
if (Dax_TBB_FOUND)
  add_executable(lhmcTBB ${headers} ${sources})
  set_target_properties(lhmcTBB PROPERTIES COMPILE_DEFINITIONS
                      "DAX_DEVICE_ADAPTER=DAX_DEVICE_ADAPTER_TBB")
  target_link_libraries(lhmcTBB ${OPENGL_LIBRARIES}
                        ${GLUT_LIBRARIES} ${GLEW_LIBRARIES} ${TBB_LIBRARIES})
endif (Dax_TBB_FOUND)

#-----------------------------------------------------------------------------
if (Dax_Cuda_FOUND)
  set(cuda_sources ${sources})
  list(REMOVE_ITEM cuda_sources Mandlebulb.cxx)
  list(REMOVE_ITEM cuda_sources Window.cxx )
  list(REMOVE_ITEM cuda_sources main.cxx )
  list(APPEND cuda_sources Mandlebulb.cu Window.cu  main.cu)
  cuda_add_executable(lhmcCuda ${headers} ${cuda_sources})
      set_target_properties(lhmcCuda PROPERTIES COMPILE_DEFINITIONS
                            "DAX_DEVICE_ADAPTER=DAX_DEVICE_ADAPTER_CUDA")
  target_link_libraries(lhmcCuda
                        ${OPENGL_LIBRARIES}
                        ${GLUT_LIBRARIES}
                        ${GLEW_LIBRARIES})
endif (Dax_Cuda_FOUND)
