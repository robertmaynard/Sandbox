# - Find ThreadingBuildingBlocks include dirs and libraries
# Use this module by invoking find_package with the form:
#  find_package(TBB
#    [REQUIRED]             # Fail with error if TBB is not found
#    )                      #
# Once done, this will define
#
#  TBB_FOUND - system has TBB
#  TBB_INCLUDE_DIRS - the TBB include directories
#  TBB_LIBRARIES - link these to use TBB
#
#  TBB_MALLOC_FOUND - system has TBB malloc library
#  TBB_MALLOC_INCLUDE_DIRS
#  TBB_MALLOC_LIBRARIES
#
#  TBB_MALLOC_PROXY_FOUND - system has TBB malloc proxy library
#  TBB_MALLOC_PROXY_INCLUDE_DIRS
#  TBB_MALLOC_PROXY_LIBRARIES
#
#
# This module reads hints about search locations from variables:
#  ENV TBB_ARCH_PLATFORM for windows only
#  ENV TBB_ROOT or just TBB_ROOT
#
#
#
# Modified by Robert Maynard from the original OGRE source
#
#-------------------------------------------------------------------
# This file is part of the CMake build system for OGRE
#     (Object-oriented Graphics Rendering Engine)
# For the latest info, see http://www.ogre3d.org/
#
# The contents of this file are placed in the public domain. Feel
# free to make use of it in any way you like.
#-------------------------------------------------------------------
#
#=============================================================================
# Copyright 2010-2012 Kitware, Inc.
# Copyright 2012      Rolf Eike Beer <eike@sf-mail.de>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)


#=============================================================================
#  FindTBB helper functions and macros
#


#===============================================
# See if we have env vars to help us find tbb
#===============================================
macro(getenv_path VAR)
   set(ENV_${VAR} $ENV{${VAR}})
   # replace won't work if var is blank
   if (ENV_${VAR})
     string( REGEX REPLACE "\\\\" "/" ENV_${VAR} ${ENV_${VAR}} )
   endif ()
endmacro(getenv_path)

#===============================================
# clear vars if the user changes them
#===============================================
macro(clear_if_changed TESTVAR)
  # test against internal check variable
  # HACK: Apparently, adding a variable to the cache cleans up the list
  # a bit. We need to also remove any empty strings from the list, but
  # at the same time ensure that we are actually dealing with a list.
  list(APPEND ${TESTVAR} "")
  list(REMOVE_ITEM ${TESTVAR} "")
  if (NOT "${${TESTVAR}}" STREQUAL "${${TESTVAR}_INT_CHECK}")
    message(STATUS "${TESTVAR} changed.")
    foreach(var ${ARGN})
      set(${var} "NOTFOUND" CACHE STRING "x" FORCE)
    endforeach(var)
  endif ()
  set(${TESTVAR}_INT_CHECK ${${TESTVAR}} CACHE INTERNAL "x" FORCE)
endmacro(clear_if_changed)

#===============================================
# Generate debug names from given release names
#===============================================
macro(get_debug_names PREFIX)
  foreach(i ${${PREFIX}})
    set(${PREFIX}_DBG ${${PREFIX}_DBG} ${i}d ${i}D ${i}_d ${i}_D ${i}_debug ${i})
  endforeach(i)
endmacro(get_debug_names)


#===============================================
# Couple a set of release AND debug libraries
#===============================================
macro(make_library_set PREFIX)
  if (${PREFIX}_REL AND ${PREFIX}_DBG)
    set(${PREFIX} optimized ${${PREFIX}_REL} debug ${${PREFIX}_DBG})
  elseif (${PREFIX}_REL)
    set(${PREFIX} ${${PREFIX}_REL})
  elseif (${PREFIX}_DBG)
    set(${PREFIX} ${${PREFIX}_DBG})
  endif ()
endmacro(make_library_set)

#===============================================
# Do the final processing for the package find.
#===============================================
macro(create_search_paths PREFIX)
  foreach(dir ${${PREFIX}_PREFIX_PATH})
    set(${PREFIX}_INC_SEARCH_PATH ${${PREFIX}_INC_SEARCH_PATH}
      ${dir}/include ${dir}/Include ${dir}/include/${PREFIX} ${dir}/Headers)
    set(${PREFIX}_LIB_SEARCH_PATH ${${PREFIX}_LIB_SEARCH_PATH}
      ${dir}/lib ${dir}/Lib ${dir}/lib/${PREFIX} ${dir}/Libs)
    set(${PREFIX}_BIN_SEARCH_PATH ${${PREFIX}_BIN_SEARCH_PATH}
      ${dir}/bin)
  endforeach(dir)
  if(ANDROID)
    set(${PREFIX}_LIB_SEARCH_PATH ${${PREFIX}_LIB_SEARCH_PATH} ${OGRE_DEPENDENCIES_DIR}/lib/${ANDROID_ABI})
  endif()
  set(${PREFIX}_FRAMEWORK_SEARCH_PATH ${${PREFIX}_PREFIX_PATH})
endmacro(create_search_paths)

#===============================================
# Do the final processing for the package find.
#===============================================
macro(findpkg_finish PREFIX)
  # skip if already processed during this run
  if (NOT ${PREFIX}_FOUND)
    if (${PREFIX}_INCLUDE_DIR AND ${PREFIX}_LIBRARY)
      set(${PREFIX}_FOUND TRUE)
      set(${PREFIX}_INCLUDE_DIRS ${${PREFIX}_INCLUDE_DIR})
      set(${PREFIX}_LIBRARIES ${${PREFIX}_LIBRARY})
      if (NOT ${PREFIX}_FIND_QUIETLY)
        message(STATUS "Found ${PREFIX}: ${${PREFIX}_LIBRARIES}")
      endif ()
    else ()
      if (NOT ${PREFIX}_FIND_QUIETLY)
        message(STATUS "Could not locate ${PREFIX}")
      endif ()
      if (${PREFIX}_FIND_REQUIRED)
        message(FATAL_ERROR "Required library ${PREFIX} not found! Install the library (including dev packages) and try again. If the library is already installed, set the missing variables manually in cmake.")
      endif ()
    endif ()

    mark_as_advanced(${PREFIX}_INCLUDE_DIR ${PREFIX}_LIBRARY ${PREFIX}_LIBRARY_REL ${PREFIX}_LIBRARY_DBG ${PREFIX}_LIBRARY_FWK)
  endif ()
endmacro(findpkg_finish)

#=============================================================================
#  Now to actually find TBB
#

# Get path, convert backslashes as ${ENV_${var}}
getenv_path(TBB_ROOT)

# get the arch, only used by windows
set(TBB_ARCH_PLATFORM $ENV{TBB_ARCH_PLATFORM})

# construct search paths
set(TBB_PREFIX_PATH ${TBB_ROOT} ${ENV_TBB_ROOT})

# redo search if prefix path changed
clear_if_changed(
  TBB_PREFIX_PATH
  TBB_LIBRARY_REL
  TBB_LIBRARY_DBG
  TBB_INCLUDE_DIR
  )

# For Windows, let's assume that the user might be using the precompiled
# TBB packages from the main website. These use a rather awkward directory
# structure (at least for automatically finding the right files) depending
# on platform and compiler, but we'll do our best to accommodate it.
# Not adding the same effort for the precompiled linux builds, though. Those
# have different versions for CC compiler versions and linux kernels which
# will never adequately match the user's setup, so there is no feasible way
# to detect the "best" version to use. The user will have to manually
# select the right files. (Chances are the distributions are shipping their
# custom version of tbb, anyway, so the problem is probably nonexistant.)
if (WIN32 AND MSVC)
  set(COMPILER_PREFIX "vc7.1")
  if (MSVC_VERSION EQUAL 1400)
    set(COMPILER_PREFIX "vc8")
  elseif(MSVC_VERSION EQUAL 1500)
    set(COMPILER_PREFIX "vc9")
  elseif(MSVC_VERSION EQUAL 1600)
    set(COMPILER_PREFIX "vc10")
  elseif(MSVC_VERSION EQUAL 1700)
    set(COMPILER_PREFIX "vc11")
  elseif(MSVC_VERSION EQUAL 1800)
    set(COMPILER_PREFIX "vc12")
  endif ()

  # for each prefix path, add ia32/64\${COMPILER_PREFIX}\lib to the lib search path
  foreach (dir ${TBB_PREFIX_PATH})
    if (CMAKE_CL_64)
      list(APPEND TBB_LIB_SEARCH_PATH ${dir}/ia64/${COMPILER_PREFIX}/lib)
      list(APPEND TBB_LIB_SEARCH_PATH ${dir}/lib/ia64/${COMPILER_PREFIX})
      list(APPEND TBB_LIB_SEARCH_PATH ${dir}/intel64/${COMPILER_PREFIX}/lib)
      list(APPEND TBB_LIB_SEARCH_PATH ${dir}/lib/intel64/${COMPILER_PREFIX})
    else ()
      list(APPEND TBB_LIB_SEARCH_PATH ${dir}/ia32/${COMPILER_PREFIX}/lib)
      list(APPEND TBB_LIB_SEARCH_PATH ${dir}/lib/ia32/${COMPILER_PREFIX})
    endif ()
  endforeach ()
endif ()

foreach (dir ${TBB_PREFIX_PATH})
  list(APPEND TBB_LIB_SEARCH_PATH ${dir}/${TBB_ARCH_PLATFORM}/lib)
  list(APPEND TBB_LIB_SEARCH_PATH ${dir}/lib/${TBB_ARCH_PLATFORM})
  list(APPEND TBB_LIB_SEARCH_PATH ${dir}/lib)
endforeach ()


set(TBB_LIBRARY_NAMES tbb)
get_debug_names(TBB_LIBRARY_NAMES)


find_path(TBB_INCLUDE_DIR
          NAMES tbb/tbb.h
          HINTS ${TBB_PREFIX_PATH}
         )

find_library(TBB_LIBRARY_REL
             NAMES ${TBB_LIBRARY_NAMES}
             HINTS ${TBB_PREFIX_PATH}
             )

find_library(TBB_LIBRARY_DBG
             NAMES ${TBB_LIBRARY_NAMES_DBG}
             HINTS ${TBB_PREFIX_PATH})

make_library_set(TBB_LIBRARY)
findpkg_finish(TBB)

if (NOT TBB_FOUND)
  return()
endif ()


# Look for TBB's malloc package
set(TBB_MALLOC_LIBRARY_NAMES tbbmalloc)
get_debug_names(TBB_MALLOC_LIBRARY_NAMES)

find_path(TBB_MALLOC_INCLUDE_DIR
          NAMES tbb/tbb.h
          PATHS ${TBB_PREFIX_PATH})

find_library(TBB_MALLOC_LIBRARY_REL
             NAMES ${TBB_MALLOC_LIBRARY_NAMES}
             PATHS ${TBB_PREFIX_PATH})

find_library(TBB_MALLOC_LIBRARY_DBG
             NAMES ${TBB_MALLOC_LIBRARY_NAMES_DBG}
             PATHS ${TBB_PREFIX_PATH})

make_library_set(TBB_MALLOC_LIBRARY)
findpkg_finish(TBB_MALLOC)

# Look for TBB's malloc proxy package
set(TBB_MALLOC_PROXY_LIBRARY_NAMES tbbmalloc_proxy)
get_debug_names(TBB_MALLOC_PROXY_LIBRARY_NAMES)

find_path(TBB_MALLOC_PROXY_INCLUDE_DIR
          NAMES tbb/tbbmalloc_proxy.h
          PATHS ${TBB_PREFIX_PATH})

find_library(TBB_MALLOC_PROXY_LIBRARY_REL
             NAMES ${TBB_MALLOC_PROXY_LIBRARY_NAMES}
             PATHS ${TBB_PREFIX_PATH})

find_library(TBB_MALLOC_PROXY_LIBRARY_DBG
             NAMES ${TBB_MALLOC_PROXY_LIBRARY_NAMES_DBG}
             PATHS ${TBB_PREFIX_PATH})

make_library_set(TBB_MALLOC_PROXY_LIBRARY)
findpkg_finish(TBB_MALLOC_PROXY)
