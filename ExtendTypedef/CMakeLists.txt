cmake_minimum_required (VERSION 2.8)
project(ExtendTypedef)

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(sources main.cxx Modify.h Functor.h)
add_executable(extend ${sources})
