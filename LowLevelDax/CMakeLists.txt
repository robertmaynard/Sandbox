cmake_minimum_required(VERSION 2.8)

project(DaxPistonExample CXX)

find_package(Dax REQUIRED)
DaxConfigureSerial(REQUIRED)
DaxConfigureTBB()
DaxConfigureCuda()

add_executable(Threshold threshold.cxx)
add_executable(RenderThreshold render_threshold.cxx)

#setup the default device adapter to be tbb
set_target_properties(Threshold PROPERTIES COMPILE_DEFINITIONS
                      "DAX_DEVICE_ADAPTER=DAX_DEVICE_ADAPTER_TBB")
set_target_properties(RenderThreshold PROPERTIES COMPILE_DEFINITIONS
                      "DAX_DEVICE_ADAPTER=DAX_DEVICE_ADAPTER_TBB")
