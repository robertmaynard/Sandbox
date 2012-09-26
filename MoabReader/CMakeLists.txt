CMAKE_MINIMUM_REQUIRED(VERSION 2.8.2 FATAL_ERROR)

project(vtkMoabReader)

find_library(MOAB_LIBRARY MOAB REQUIRED)

# If SiMBA is installed, use SiMBA/bld/install/include for MOAB_INCLUDE_DIRS
find_path(MOAB_INCLUDE_DIRS Core.hpp REQUIRED)
include_directories(${MOAB_INCLUDE_DIRS})


find_package(ParaView REQUIRED)
include(${ParaView_USE_FILE})
include_directories(${PARAVIEW_INCLUDE_DIRS})

add_library(vtkMoabReader
            vtkMoabReader.cxx
            SimpleMoab.h
            )

message(STATUS "VTK_LIBRARIES: ${VTK_LIBRARIES}")
target_link_libraries(vtkMoabReader
                      ${VTK_LIBRARIES}
                      ${MOAB_LIBRARY})
