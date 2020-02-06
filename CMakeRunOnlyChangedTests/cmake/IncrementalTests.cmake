

#The current issue is that tests that fail
#can't be re-run as we consider them to 'finished'
#this can be fixed inside cmake since we can
#say tests that hash properly, but are recorded
#in 'rerun-failed' still need to be run

function( mark_as_inc_test test_name exec_name )

####
# Setup the hash files as test fixture setups

if(NOT TEST ${exec_name}_checkHash)
	set(hash_command ${CMAKE_COMMAND}
		-DEXEC_PATH=$<TARGET_FILE:${exec_name}>
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_hash.cmake
		)
	add_test(NAME ${exec_name}_checkHash COMMAND ${hash_command})
	set_tests_properties(${exec_name}_checkHash
			PROPERTIES FIXTURES_SETUP ${exec_name}_hash_changed)

	set_property(TEST ${exec_name}_checkHash PROPERTY
			SKIP_REGULAR_EXPRESSION "SKIP: Nothing has changed"
		)
endif()

set_tests_properties(${test_name}  PROPERTIES
					 FIXTURES_REQUIRED ${exec_name}_hash_changed)

endfunction()
