
cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(DebugLinkFile)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
include(DebugLink)

add_library(sharedLib SHARED functions.cpp)
set_property(TARGET sharedLib PROPERTY VERSION "1.0.0")
set_property(TARGET sharedLib PROPERTY SOVERSION "1")

add_executable(example main.cpp)

target_link_libraries(example PRIVATE sharedLib)
target_compile_features(example PRIVATE cxx_std_11)

install_debug_info(TARGETS sharedLib example
                   DEBUG_POSTFIX ".debug"
                   RUNTIME_DESTINATION bin
                   LIBRARY_DESTINATION lib
                   ARCHIVE_DESTINATION lib
                   )
