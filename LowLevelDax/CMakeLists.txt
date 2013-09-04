cmake_minimum_required(VERSION 2.8)

project(DaxPistonExample CXX)

find_package(Dax REQUIRED)
DaxConfigureSerial(REQUIRED)
DaxConfigureTBB()
DaxConfigureCuda()

add_executable(Threshold threshold.cxx)
#setup the default device adapter to be tbb
set_target_properties(Threshold PROPERTIES COMPILE_DEFINITIONS
                      "DAX_DEVICE_ADAPTER=DAX_DEVICE_ADAPTER_TBB")


find_package(OpenGL)
find_package(GLEW)
find_package(GLUT)

if(OPENGL_FOUND AND GLEW_FOUND AND GLUT_FOUND)
  #setup the rendering example if we have opengl, glut and glew
  add_executable(RenderThreshold render_threshold.cxx)
  target_link_libraries(RenderThreshold ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})
  set_target_properties(RenderThreshold PROPERTIES COMPILE_DEFINITIONS
                      "DAX_DEVICE_ADAPTER=DAX_DEVICE_ADAPTER_TBB")

endif()


