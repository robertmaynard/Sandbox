cmake_minimum_required(VERSION 2.8)
project(ShibokenItem)
find_package(Shiboken REQUIRED)

set(srcs
    ShibokenItem.cxx
    ShibokenManager.cxx)

add_library(ShibokenItemLibs SHARED ${srcs})

set(type_system_file "${CMAKE_CURRENT_SOURCE_DIR}/typesystem.xml")
set(global_header_file "${CMAKE_CURRENT_SOURCE_DIR}/global.h")
set(binding_file "${CMAKE_CURRENT_BINARY_DIR}/binding.txt")

set(generated_sources "${CMAKE_CURRENT_BINARY_DIR}/Example/a_b_ShibokenItem_wrapper.cpp")
set(generated_sources "${CMAKE_CURRENT_BINARY_DIR}/Example/a_b_ShibokenManager_wrapper.cpp")


set(typesystem_includes ${ShibokenItem_SOURCE_DIR}
                        ${ShibokenItem_BINARY_DIR})

#shiboken wants a colon separated list not a semicolon.
string(REPLACE ";" ":" typesystem_formatted_includes "${typesystem_includes}")

#generate the binding file which is the last step as it needs
#to have the sources already created
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/binding.txt.in"
               "${CMAKE_CURRENT_BINARY_DIR}/binding.txt" @ONLY)

#now we can actuall call shiboken
add_custom_command(OUTPUT ${generated_sources}
COMMAND ${SHIBOKEN_BINARY} --project-file=${binding_file}
DEPENDS ShibokenItemLibs ${type_system_file}
        ${global_header_file}
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
COMMENT "Running generator for 'ShibokenItemLibs' python bindings"
)

include_directories( ${CMAKE_CURRENT_BINARY_DIR}
                     ${SHIBOKEN_PYTHON_INCLUDE_DIR}
                     ${SHIBOKEN_INCLUDE_DIR}
                     ${typesystem_includes} )

set(wrapped_name ShibokenItemLibsPython)
add_library(${wrapped_name} MODULE ${generated_sources})
set_property(TARGET ${wrapped_name} PROPERTY PREFIX "")
target_link_libraries(${wrapped_name}
                      ShibokenItemLibs
                      ${SHIBOKEN_LIBRARY}
                      ${SHIBOKEN_PYTHON_LIBRARIES})

#todo we need to install these wrappings properly
