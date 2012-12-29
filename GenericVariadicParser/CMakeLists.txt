cmake_minimum_required (VERSION 2.8)
project(GenericVariadicParser)

include(CMake/VariadicSupport.cmake)

#runs the try compiles to determine what the current compiler handles
#also sepcifies the compiler options, and configures the headers to
#setup a tuple class, and helpers to mimic variadic templates using Boost
#preprocessor magic if required.
setupVariadicSupport(utility)


find_package(Boost REQUIRED COMPONENTS timer system)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${GenericVariadicParser_SOURCE_DIR})

set(sources main.cxx ParserBase.h DefaultParser DerivedParser.h)
add_executable(parse ${sources})
target_link_libraries(parse ${Boost_LIBRARIES})
