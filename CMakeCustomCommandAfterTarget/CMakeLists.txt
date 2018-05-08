
cmake_minimum_required(VERSION 3.3)
project(CMakeCustomCommandAfterTarget )

add_library(A STATIC impl1.cpp)


set(input_file ${CMAKE_CURRENT_SOURCE_DIR}/resource.in)
set(output_file ${CMAKE_CURRENT_BINARY_DIR}/resource.h)

add_custom_command(
  OUTPUT "${output_file}"
  COMMAND ${CMAKE_COMMAND} -E copy ${input_file} ${output_file}
  VERBATIM
  DEPENDS A
  COMMENT "Generating resources after Target A"
  )

add_custom_target(
    generate-resource
    DEPENDS ${output_file}
    )

add_executable(Foo foo.cpp)
add_dependencies(Foo generate-resource)

target_link_libraries(Foo PUBLIC A)
target_include_directories(Foo PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
