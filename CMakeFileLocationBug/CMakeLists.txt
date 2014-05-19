cmake_minimum_required(VERSION 2.8)
project(Example)

set(gen_path "${CMAKE_CURRENT_BINARY_DIR}//./foo")
set(gen_file_c "${gen_path}/foo.cxx")
set(gen_file_h "${gen_path}/foo.h")

add_custom_command(
  OUTPUT ${gen_path} ${gen_file_h}
  # Make sure the output directory exists before trying to write to it.
  COMMAND ${CMAKE_COMMAND} -E make_directory "${gen_path}"
  COMMAND ${CMAKE_COMMAND} -E touch "${gen_file_h}")

add_custom_command(
  OUTPUT ${gen_file_c}
  DEPENDS ${gen_file_h}
  COMMAND ${CMAKE_COMMAND} -E touch "${gen_file_c}")


add_library(Example main.cxx ${gen_file_c})
