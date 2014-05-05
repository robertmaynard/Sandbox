cmake_minimum_required(VERSION 2.8.12)

project(ImportTarget)

set(import_name
    "${CMAKE_STATIC_LIBRARY_PREFIX}Test.${CMAKE_STATIC_LIBRARY_SUFFIX}")
add_library(imported_target STATIC IMPORTED )
set_target_properties(imported_target PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/${import_name}"
                IMPORTED_LINK_INTERFACE_LIBRARIES
                                  "${CMAKE_CURRENT_SOURCE_DIR}/${import_name}")

add_library(my_lib SHARED lib.cxx)

add_library(my_next_lib SHARED lib2.cxx)

target_link_libraries(my_lib PRIVATE imported_target)
target_link_libraries(my_next_lib PUBLIC my_lib)