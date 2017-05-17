
cmake_minimum_required(VERSION 3.3)
project(CMakeInterfaceChangesLibrary )


add_library(I INTERFACE)

add_library(IA STATIC impl1.cpp)
add_library(IB STATIC impl2.cpp)


set(has_iface "$<BOOL:$<TARGET_PROPERTY:USE_IFACE_A>>")
target_link_libraries(I
  INTERFACE
  $<${has_iface}:IA>
  $<$<NOT:${has_iface}>:IB>
  )

add_executable(Foo foo.cpp)
target_link_libraries(Foo PUBLIC I)

#with neither set you will link to IB
set_target_properties(Foo PROPERTIES USE_IFACE_A 1)  #uncomment this to link to IA
#set_target_properties(Foo PROPERTIES USE_IFACE_A 0)  #uncomment this to link to IB

