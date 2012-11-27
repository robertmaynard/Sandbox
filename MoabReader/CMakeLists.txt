CMAKE_MINIMUM_REQUIRED(VERSION 2.8.2 FATAL_ERROR)

project(vtkMoabReader)

find_library(MOAB_LIBRARY MOAB REQUIRED)

# If SiMBA is installed, use SiMBA/bld/install/include for MOAB_INCLUDE_DIRS
find_path(MOAB_INCLUDE_DIRS Core.hpp REQUIRED)
include_directories(${MOAB_INCLUDE_DIRS})


find_package(ParaView REQUIRED)
include(${ParaView_USE_FILE})
include_directories(${PARAVIEW_INCLUDE_DIRS})

set(headers
  DataSetConverter.h
  detail/CellTypeToType.h
  detail/ContinousCellInfo.h
  detail/LoadGeometry.h
  detail/MixedCellConnectivity.h
  detail/ReadSparseTag.h
  detail/ReduceSpectralMesh.h
  SimpleMoab.h
  )

add_paraview_plugin(vtkMoabReaderPlugin "5.0"
   SERVER_MANAGER_XML vtkMoabReader.xml
   SERVER_MANAGER_SOURCES vtkMoabReader.cxx
   SERVER_SOURCES ${headers}
   GUI_RESOURCE_FILES vtkMoabReaderGUI.xml)

target_link_libraries(vtkMoabReaderPlugin
                      ${MOAB_LIBRARY})
