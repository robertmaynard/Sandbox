cmake_minimum_required (VERSION 2.8)
project(GenericVariadicParser)

#enable c++11 support
list( APPEND CMAKE_CXX_FLAGS "-std=c++0x")

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${GenericVariadicParser_SOURCE_DIR})

set(sources main.cxx BaseParser.h DerivedParser.h)
add_executable(parse ${sources})
