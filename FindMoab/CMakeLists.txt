cmake_minimum_required (VERSION 2.8)
project(MoabExample)

set(CMAKE_MODULE_PATH  ${MoabExample_SOURCE_DIR})

#set these to true if you want netcdf or hdf5 support
#they have to be set before you try to find the MOAB package
set(MOAB_WITH_NETCDF FALSE)
set(MOAB_WITH_HDF5 FALSE)
find_package(MOAB REQUIRED)
include_directories(${MOAB_INCLUDE_DIR})

set(sources main.cxx)
add_executable(moabExample ${sources})
target_link_libraries(moabExample ${MOAB_LIBRARIES})
