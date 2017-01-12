cmake_minimum_required (VERSION 3.3)
project(ExplicitTemplate)

set(sources template.cxx)

add_library(examples SHARED ${sources} )
target_include_directories(examples PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
