
#we are toolchain file so set the name and compiler
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_SYSTEM_NAME Linux)
  # set(CMAKE_INCLUDE_FLAG_CXX "--include=")
endif()

#CMake will try to include the file Platform/${CMAKE_SYSTEM_NAME}-${CMAKE_CXX_COMPILER_ID}-CXX
#so we need to make sure ours is included instead. These just are the overrides
#so you only need to specify non default values
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
