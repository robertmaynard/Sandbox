
#we are toolchain file so set the name and compiler
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_SYSTEM_NAME Linux)
  set(CMAKE_C_COMPILER   /Users/robert/Work/my_own-gcc)
endif()

#set(CMAKE_FIND_ROOT_PATH ${CMAKE_SOURCE_DIR})

#here is the fun part compiler verification only looks at CMAKE_CXX_FLAGS
#and than cmake will clear both CMAKE_CXX_FLAGS and CMAKE_CXX_FLAGS_INIT,
#but after that is done if it detects a CMAKE_USER_MAKE_RULES_OVERRIDE file
#it will re-parse that file and set CMAKE_CXX_FLAGS to be CMAKE_CXX_FLAGS_INIT
#it is done all processing. So to get this file to be a toolchain and a
#override we set both
set(CMAKE_CXX_FLAGS_INIT "-g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_INIT}")

