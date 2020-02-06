cmake_minimum_required(VERSION 3.16)
project(CTest_inc_)

add_library(libA SHARED libA.cxx)
add_library(libB SHARED libB.cxx)
add_library(libC SHARED libC.cxx)

target_compile_features(libA PUBLIC cxx_std_14)
target_compile_features(libB PUBLIC cxx_std_14)
target_compile_features(libC PUBLIC cxx_std_14)

add_executable(execA mainA.cxx)
add_executable(execB mainB.cxx)
add_executable(execC mainC.cxx)

target_link_libraries(execA PRIVATE libA)
target_link_libraries(execB PRIVATE libB)
target_link_libraries(execC PRIVATE libC)

#Note: We need to have a test cleanup that runs
# and if all tests have passed, we can delete
# the re-run marker file

enable_testing()
add_test(NAME test1 COMMAND execA)
add_test(NAME test2 COMMAND execB)
add_test(NAME test3 COMMAND execC)
add_test(NAME test3b COMMAND execC)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/IncrementalTests.cmake)
mark_as_inc_test(test1 execA)
mark_as_inc_test(test2 execB)
mark_as_inc_test(test3 execC)
mark_as_inc_test(test3b execC)
