cmake_minimum_required(VERSION 2.8.11)

project(Example)

find_package(CUDA REQUIRED)
find_package(ZLIB REQUIRED)

set(srcs foo.cu main.cxx)

cuda_add_library(example STATIC ${srcs})
target_link_libraries(example ${ZLIB_LIBRARIES})
target_include_directories(example PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}
                           ${ZLIB_INCLUDE_DIRS})
