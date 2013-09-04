cmake_minimum_required(VERSION 2.8)

project(DaxPistonExample CXX)

find_package(Dax REQUIRED)
DaxConfigureSerial(REQUIRED)
DaxConfigureTBB()
DaxConfigureCuda()

add_executable(ExampleThreshold threshold.cxx)

#setup the default device adapter to be tbb
set_target_properties(ExampleThreshold PROPERTIES COMPILE_DEFINITIONS
                      "DAX_DEVICE_ADAPTER=DAX_DEVICE_ADAPTER_TBB")
