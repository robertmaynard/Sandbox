CMAKE_MINIMUM_REQUIRED(VERSION 2.8.2 FATAL_ERROR)

project(vtkMoabReader)

find_library(MOAB_LIBRARY MOAB REQUIRED)

# If SiMBA is installed, use SiMBA/bld/install/include for MOAB_INCLUDE_DIRS
find_path(MOAB_INCLUDE_DIRS Core.hpp REQUIRED)
include_directories(${MOAB_INCLUDE_DIRS})


FIND_PACKAGE (ParaView REQUIRED)
include (${PARAVIEW_USE_FILE})

add_executable(vtkMoabReader
               vtkMoabReader.cxx
               SimpleMoab.h
               )

target_link_libraries(vtkMoabReader
                    ${VTK_LIBS}
                    ${MOAB_LIBRARY})
