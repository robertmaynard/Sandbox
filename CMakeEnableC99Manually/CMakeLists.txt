cmake_minimum_required (VERSION 2.8.8)
project(Example C CXX)


function(set_language_flags target)
  if(CMAKE_VERSION LESS 3.1)
    #this is where we get a bit more complex as we need to iterate the SOURCES
    #and set compile flags on a per file basis as we cant specify per language
    #flags on targets in older cmake versions
    get_target_property(sources ${target} SOURCES)
    foreach(file IN LISTS sources)
      get_source_file_property(lang ${file} LANGUAGE)
      message(STATUS "file ${file} lang: ${lang}")
      if("${lang}" STREQUAL "CXX")
        set_property(SOURCE "${file}" PROPERTY COMPILE_FLAGS "-std=c++11")
      elseif("${lang}" STREQUAL "C")
        set_property(SOURCE "${file}" PROPERTY COMPILE_FLAGS "-std=c99")
      endif()

    endforeach()

  else()
    set_property(TARGET ${target} PROPERTY C_STANDARD 99)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD 11)
  endif()
endfunction()


add_executable(Example main.cpp static.c)
set_language_flags(Example)

