cmake_minimum_required(VERSION 2.8.8)

project(DumpTags)

find_library(MOAB_LIBRARY MOAB REQUIRED)

# If SiMBA is installed, use SiMBA/bld/install/include for MOAB_INCLUDE_DIRS
find_path(MOAB_INCLUDE_DIRS Core.hpp REQUIRED)
include_directories(${MOAB_INCLUDE_DIRS})


add_executable(DUMP
               main.cxx
               )

target_link_libraries(DUMP ${MOAB_LIBRARY})

