

find_program(objcopy_cmd NAMES objcpy dsymutil REQUIRED)
add_executable(objcpy IMPORTED)
set_property(TARGET objcpy PROPERTY IMPORTED_LOCATION ${objcopy_cmd})

find_program(strip_cmd strip REQUIRED)
add_executable(strip IMPORTED)
set_property(TARGET strip PROPERTY IMPORTED_LOCATION ${strip_cmd})

function(install_debug_info )
  set(options "")
  set(oneValueArgs DEBUG_POSTFIX RUNTIME_DESTINATION LIBRARY_DESTINATION ARCHIVE_DESTINATION)
  set(multiValueArgs TARGETS "")
  cmake_parse_arguments(debinfo "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

  # first we setup the install paths for everything past to us

  set(install_locations)
  if(debinfo_RUNTIME_DESTINATION)
    list(APPEND install_locations RUNTIME DESTINATION ${debinfo_RUNTIME_DESTINATION})
  endif()
  if(debinfo_LIBRARY_DESTINATION)
    list(APPEND install_locations LIBRARY DESTINATION ${debinfo_LIBRARY_DESTINATION})
  endif()
  if(debinfo_ARCHIVE_DESTINATION)
    list(APPEND install_locations ARCHIVE DESTINATION ${debinfo_ARCHIVE_DESTINATION})
  endif()

  install(TARGETS ${debinfo_TARGETS}
          ${install_locations}
          ${debinfo_UNPARSED_ARGUMENTS}
          )

  set(postfix ".debug")
  if(debinfo_DEBUG_POSTFIX)
    set(postfix ${debinfo_DEBUG_POSTFIX})
  endif()

  foreach(target ${debinfo_TARGETS})
    # we only want to support a white list of
    # types: EXECUTABLE, SHARED_LIBRARY, and MODULE_LIBRARY

    get_target_property(libtype ${target} TYPE)
    get_target_property(isbundle ${target} BUNDLE)

    set(dest )
    if(${libtype} STREQUAL "SHARED_LIBRARY")
      set(dest ${debinfo_LIBRARY_DESTINATION})
    elseif(${libtype} STREQUAL "MODULE_LIBRARY")
      set(dest ${debinfo_LIBRARY_DESTINATION})
    elseif(${libtype} STREQUAL "EXECUTABLE" AND NOT isbundle)
      set(dest ${debinfo_RUNTIME_DESTINATION})
    else()
      continue()
    endif()

    if(APPLE)
      add_custom_command(TARGET ${target} POST_BUILD
        COMMAND objcpy -f --out=$<TARGET_FILE:${target}>${postfix} $<TARGET_FILE:${target}>
        COMMAND strip -S $<TARGET_FILE:${target}>
      )
    else()
      add_custom_command(TARGET ${target} POST_BUILD
        COMMAND objcpy --only-keep-debug $<TARGET_FILE:${target}> $<TARGET_FILE:${target}>${postfix}
        COMMAND strip --strip-debug $<TARGET_FILE:${target}>
        COMMAND objcpy --add-gnu-debuglink=$<TARGET_FILE:${target}>${postfix} $<TARGET_FILE:${target}>
      )
    endif()

    # todo grab the real desitnation from the args past in
    install(FILES $<TARGET_FILE:${target}>${postfix}
            DESTINATION ${dest}
            )

  endforeach()
endfunction()
