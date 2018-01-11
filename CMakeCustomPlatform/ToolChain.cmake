
#we are toolchain file so set the name and compiler
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_SYSTEM_NAME Generic)
endif()

# Ask the intial try compile to be a static library instead of an executable
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_C_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/fake_compiler)

#CMake will try to include the file Platform/${CMAKE_SYSTEM_NAME}-${CMAKE_CXX_COMPILER_ID}-CXX
#so we need to make sure ours is included instead. These just are the overrides
#so you only need to specify non default values
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# We need to make sure that CMake's compiler verification step
# has our form of how the compiler expects the output flag and
# input flag to be.
set(CMAKE_C_COMPILE_OBJECT  "<CMAKE_C_COMPILER> --compile=<SOURCE> <DEFINES> <INCLUDES> <FLAGS> --output=<OBJECT>")


