cmake_minimum_required(VERSION 2.8.11)

project(Example)

find_package(CUDA REQUIRED)
set(srcs foo.cu)

cuda_add_library(example STATIC ${srcs})

#fake an import library
add_library(zlib SHARED IMPORTED)
set_property(TARGET zlib APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(zlib PROPERTIES
  IMPORTED_LOCATION_DEBUG "/Users/robert/zlib/libz.dylib"
  IMPORTED_SONAME_RELEASE "/Users/robert/zlib/libz.dylib"
  )

target_link_libraries(example ${CUDA_LIBRARIES} zlib )
target_include_directories(example
                           PUBLIC
                            ${CMAKE_CURRENT_SOURCE_DIR})

#add an executable jut for testing
add_executable(examplebin main.cxx)
target_link_libraries(examplebin PRIVATE example)