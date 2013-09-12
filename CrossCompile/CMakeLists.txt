cmake_minimum_required (VERSION 2.6)

#setup the override file to match the toolchain
#needs to be done before the project call
get_filename_component(override_name ${CMAKE_TOOLCHAIN_FILE} NAME)
set(CMAKE_USER_MAKE_RULES_OVERRIDE ${override_name})

project(CrossCompile)




message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_INIT: ${CMAKE_CXX_FLAGS_INIT}")
message(STATUS "_CMAKE_TOOLCHAIN_PREFIX: ${_CMAKE_TOOLCHAIN_PREFIX}")

add_executable(CrossCompile main.cpp)

try_run(run_result compile_result
    ${CrossCompile_BINARY_DIR}
    ${CrossCompile_SOURCE_DIR}/main.cpp)
