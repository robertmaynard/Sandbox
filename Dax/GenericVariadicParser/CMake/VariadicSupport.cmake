
#This function does the following:
#
# Determine if we have c++11, c++0x support.
# If so enable thos compilation options, otherwise
# fallback to boost.
#
# Configure a header that defines helper methods
# so that we can easily write fake pre processor
# variadic templates if using boost
#
# We also configure a header to setup in the
# namespace that user provided a tuple
# class and all related helpers
#
# We save into Variadic_CXX_FLAGS the flags required
# to be added to CMAKE_CXX_FLAGS that will allow proper compilation
# of variadic support
#
# When falling back to boost whill define
# Variadic_Requires_Boost to TRUE so that
# the including project can make sure
# to find and include Boost

function(setupVariadicSupport tuple_new_namespace)

  set(compiler_type 0)
  set(c++11_compiler 1)
  set(c++0x_compiler 2)
  set(c++Boost_compiler 3)

  set(Variadic_Requires_Boost FALSE PARENT_SCOPE)
  set(VARIADIC_SUPPORT_FOUND FALSE)

  #check if we have c++11 support
  if(NOT ${VARIADIC_SUPPORT_FOUND})
    try_compile(VARIADIC_SUPPORT_FOUND
      ${PROJECT_BINARY_DIR}/CMakeTmp
      ${PROJECT_SOURCE_DIR}/CMake/variadic_11.cxx
      COMPILE_DEFINITIONS "-std=c++11"
      OUTPUT_VARIABLE output
      )
    if(${VARIADIC_SUPPORT_FOUND})
      set(compiler_type ${c++11_compiler})
      set(Variadic_CXX_FLAGS "-std=c++11" PARENT_SCOPE)
    endif()
  endif()

  #check if we have c++0x support in tr1 namespace
  if(NOT ${VARIADIC_SUPPORT_FOUND})
    try_compile(VARIADIC_SUPPORT_FOUND
    ${PROJECT_BINARY_DIR}/CMakeTmp
    ${PROJECT_SOURCE_DIR}/CMake/variadic_0x.cxx
    COMPILE_DEFINITIONS "-std=c++0x"
    OUTPUT_VARIABLE output
    )
    if(${VARIADIC_SUPPORT_FOUND})
      set(compiler_type ${c++0x_compiler})
      set(Variadic_CXX_FLAGS "-std=c++0x" PARENT_SCOPE)
    endif()
  endif()

  #fall back to using boost
  if(NOT ${VARIADIC_SUPPORT_FOUND})
    set(compiler_type ${c++Boost_compiler})
    set(Variadic_Requires_Boost TRUE PARENT_SCOPE)
    set(Variadic_CXX_FLAGS "-std=c++0x")
  endif()

  configure_file(
    ${PROJECT_SOURCE_DIR}/CMake/tuple.hpp.in
    ${PROJECT_BINARY_DIR}/@tuple_new_namespace@_tuple.hpp
    @ONLY)

  include_directories(${PROJECT_BINARY_DIR})
endfunction(setupVariadicSupport)

