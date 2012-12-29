cmake_minimum_required (VERSION 2.8)
project(GenericVariadicParser)

#enable c++11 support
list( APPEND CMAKE_CXX_FLAGS "-std=c++0x")

find_package(Boost REQUIRED COMPONENTS timer system tuple)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${GenericVariadicParser_SOURCE_DIR})

set(sources main.cxx ParserBase.h DefaultParser DerivedParser.h)
add_executable(parse ${sources})
target_link_libraries(parse ${Boost_LIBRARIES})
