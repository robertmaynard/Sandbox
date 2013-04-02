cmake_minimum_required(VERSION 2.8.10)
project(DependSideEffect C)
add_library(A a.c)
add_custom_command(
 TARGET A POST_BUILD
 COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/a.c a.txt
 )
add_custom_command(
 OUTPUT b.txt
 DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/a.txt
 COMMAND cp a.txt b.txt
 )
add_custom_target(B ALL DEPENDS b.txt)
add_dependencies(B A)