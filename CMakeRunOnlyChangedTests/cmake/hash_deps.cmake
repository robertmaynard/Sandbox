

file(GET_RUNTIME_DEPENDENCIES
  RESOLVED_DEPENDENCIES_VAR resolved_libs
  UNRESOLVED_DEPENDENCIES_VAR unresolved_libs
  EXECUTABLES "${EXEC_PATH}"
  )

file(SHA1 "${EXEC_PATH}" sha1s)
foreach(file IN LISTS resolved_libs )
  file(SHA1 "${file}" file_hash)
  message(STATUS "${file}: ${file_hash}")
  list(APPEND sha1s ${file_hash})
endforeach()

list(SORT sha1s)
string(SHA1 new_hash "${sha1s}")

set(has_changed TRUE)

#Need to now read the last hash and compare it to previous
if(EXISTS "${EXEC_PATH}.dep_hash")
	file(READ "${EXEC_PATH}.dep_hash" old_hash)

	if(new_hash STREQUAL old_hash)
		set(has_changed FALSE)
	endif()

	if(NOT has_changed AND EXISTS "${EXEC_PATH}.rerun")
		# If we somehow rebuild a target but nothing has
		# changed we can safely prune it from the
		# eligable set
		file(REMOVE "${EXEC_PATH}.rerun")
	endif()
endif()

message(STATUS "new_hash: ${new_hash} \n old_hash: ${old_hash} \n has_changed:${has_changed}")
if(has_changed)
	file(WRITE "${EXEC_PATH}.dep_hash" "${new_hash}")
	file(WRITE "${EXEC_PATH}.rerun" "1")
endif()
