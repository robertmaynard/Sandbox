cmake_minimum_required(VERSION 2.8)

project(ParticleDemo CXX)

find_package(Dax REQUIRED)
DaxConfigureSerial(REQUIRED)
DaxConfigureTBB()
DaxConfigureCuda()

find_package(OpenGL)
find_package(GLEW)
find_package(GLUT)

if(OPENGL_FOUND AND GLEW_FOUND AND GLUT_FOUND)
    #setup the rendering example if we have opengl, glut and glew
  add_executable(Particles particles.cxx)
  target_link_libraries(Particles ${TBB_LIBRARIES} ${OPENGL_LIBRARIES}
                                  ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})
  set_target_properties(Particles PROPERTIES COMPILE_DEFINITIONS
                      "DAX_DEVICE_ADAPTER=DAX_DEVICE_ADAPTER_TBB")

  if(Dax_Cuda_FOUND)
    cuda_add_executable(CudaParticles particles.cu)
    target_link_libraries(CudaParticles ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES}
                                        ${GLEW_LIBRARIES})
    set_target_properties(CudaParticles PROPERTIES COMPILE_DEFINITIONS
                        "DAX_DEVICE_ADAPTER=DAX_DEVICE_ADAPTER_CUDA")
  endif()


endif()


