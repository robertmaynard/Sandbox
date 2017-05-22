# Copyright 2017 Kitware, Inc.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

