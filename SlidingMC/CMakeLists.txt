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

project(SlidingContour CXX)

find_package(Dax REQUIRED)
DaxConfigureSerial()
DaxConfigureTBB(REQUIRED)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#-----------------------------------------------------------------------------
set(headers
  ArgumentsParser.h
  SlidingContour.h
  )

set(sources
  ArgumentsParser.cxx
  SlidingContour.cxx
  main.cxx
  )

#-----------------------------------------------------------------------------
if (Dax_TBB_FOUND)
  add_executable(slidingTBB ${headers} ${sources})
  set_target_properties(slidingTBB PROPERTIES COMPILE_DEFINITIONS
                      "DAX_DEVICE_ADAPTER=DAX_DEVICE_ADAPTER_TBB")
  target_link_libraries(slidingTBB
                        ${TBB_LIBRARIES}
                        vtkIOImage
                        vtkIOXML
                        vtkFiltersCore)
endif (Dax_TBB_FOUND)
