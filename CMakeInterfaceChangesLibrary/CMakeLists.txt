
cmake_minimum_required(VERSION 3.3)
project(CMakeInterfaceChangesLibrary )


add_library(I INTERFACE)

add_library(IA STATIC impl1.cpp)
set_target_properties(IA PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(IB STATIC impl2.cpp)


set(has_iface "$<BOOL:$<TARGET_PROPERTY:POSITION_INDEPENDENT_CODE>>")
target_link_libraries(I
  INTERFACE
  $<${has_iface}:IA>
  $<$<NOT:${has_iface}>:IB>
  )

add_executable(Foo foo.cpp)
target_link_libraries(Foo PUBLIC I) #links to IB

add_library(Bar SHARED bar.cpp)
target_link_libraries(Bar PUBLIC I) #links to IA
