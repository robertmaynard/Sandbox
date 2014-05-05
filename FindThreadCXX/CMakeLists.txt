
cmake_minimum_required(VERSION 2.8.11)

project(CMakeFindThreads CXX)

find_package(Threads)

add_executable(Example main.cxx)

target_link_libraries(Example ${CMAKE_THREAD_LIBS_INIT})


