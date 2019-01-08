cmake_minimum_required(VERSION 3.5)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH /tmp/c)

message("CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH}")
message("CMAKE_SYSTEM_PREFIX_PATH ${CMAKE_SYSTEM_PREFIX_PATH}")
message("CMAKE_SYSTEM_INCLUDE_PATH ${CMAKE_SYSTEM_INCLUDE_PATH}")

find_path(X_DIR x)
find_path(Y_DIR y PATH_SUFFIXES ../test/)
find_path(Z_DIR z)
find_path(W_DIR w PATHS include)

foreach(d X Y Z W)
    message("${d}_DIR : ${${d}_DIR}")
endforeach()

# - - -
# $ tree /tmp/c
# /tmp/c
# ├── include
# │   ├── w
# │   └── z
# ├── test
# │   └── y
# └── x

# $ cmake .
# X_DIR : X_DIR-NOTFOUND
# Y_DIR : Y_DIR-NOTFOUND
# Z_DIR : /tmp/c/include
# W_DIR : W_DIR-NOTFOUND
