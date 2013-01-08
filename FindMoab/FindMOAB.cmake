# - Try to find MOAB headers and libraries
#
# Usage of this module as follows:
#
#     find_package(MOAB)
#
# Variables used by this module, they can change the default behavior and need
# to be set before calling find_package:
#
#  MOAB_ROOT_DIR  Set this variable to the root installation of
#                 MOAB if the module has problems finding
#                 the proper installation path.
#
# MOAB_WITH_NETCDF Set this variable to state that MOAB was built
#                  with netcdf support enabled, so we need to add the netcdf
#                  library to the link line
#
# MOAB_WITH_HDF5 Set this variable to state that MOAB was built
#                with hdf5 support enabled, so we need to add the hdf5
#                library to the link line
#
#
#
# Search Variables defined by this module:
#
#  MOAB_FOUND              System has MOAB libs/headers
#  MOAB_INCLUDE_DIR        The location of MOAB headers
#  MOAB_LIBRARIES          The moab library to link too
#

find_path(MOAB_INCLUDE_DIR
  NAMES Core.hpp
  HINTS
    ENV MOAB_DIR
    ${MOAB_ROOT_DIR}
  PATH_SUFFIXES
    include/moab include
  )

find_library(MOAB_LIBRARY
  NAMES MOAB Moab
  HINTS
    ENV MOAB_DIR
    ${MOAB_ROOT_DIR}
  PATH_SUFFIXES
    lib/moab lib
)

if (MOAB_LIBRARY)

  list(APPEND MOAB_LIBRARIES ${MOAB_LIBRARY})

  if(MOAB_WITH_NETCDF)
    find_package(NetCDF)
    list(APPEND MOAB_LIBRARIES
        ${NetCDF_LIBRARIES})
  endif()

  if(MOAB_WITH_HDF5)
    #TODO: determine what exact bindings to HDF5 moab uses.
    #are they using the C library, C++ library or high level C bindings
    find_package(HDF5 COMPONENTS C CXX HL)
    list(APPEND MOAB_LIBRARIES
          ${HDF5_C_LIBRARIES}
          ${HDF5_CXX_LIBRARIES}
          ${HDF5_HL_LIBRARIES})
  endif()

endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MOAB DEFAULT_MSG
                                  MOAB_LIBRARIES MOAB_INCLUDE_DIR)

mark_as_advanced(MOAB_INCLUDE_DIR MOAB_LIBRARIES)

