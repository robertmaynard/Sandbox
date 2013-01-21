cmake_minimum_required (VERSION 2.8)
project(ExternC)

set(sources non_extern.cxx)
set(extern_sources extern.cxx)

add_library(example_lib SHARED ${sources} )
add_library(example_c_lib SHARED ${extern_sources} )
