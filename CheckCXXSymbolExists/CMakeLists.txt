cmake_minimum_required(VERSION 2.8)
project(Example)

 list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
# look for various headers and functions
include(CheckCXXSymbolExists2)
#set(CMAKE_REQUIRED_FLAGS "-std=c++11")
check_cxx_symbol_exists(std::isinf "cmath" VTK_HAS_STD_ISINF)
message(STATUS "VTK_HAS_STD_ISINF ${VTK_HAS_STD_ISINF}")

set(srcs main.cxx)
add_executable(Example ${srcs})