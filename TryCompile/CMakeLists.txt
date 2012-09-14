cmake_minimum_required (VERSION 2.8)
project(Example)

include(CMake/SharedPtr.cmake)

determineSharedPtrType(SHARED_TYPE SHARED_TYPE_HEADER)

message(STATUS "${SHARED_TYPE_HEADER}")


#if the shared_ptr isn't able to be
#found in c++ or c++ tr1 we include boost
if(${SHARED_TYPE_BOOST_TRUE})
  find_package(Boost)
endif()

configure_file(
  ${PROJECT_SOURCE_DIR}/CMake/shared.h.in
  ${PROJECT_BINARY_DIR}/slctk/shared.h
  @ONLY)


include_directories(${PROJECT_BINARY_DIR})

set(sources main.cxx)
add_executable(example ${sources})