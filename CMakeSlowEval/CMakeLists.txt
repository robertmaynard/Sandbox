cmake_minimum_required(VERSION 2.8.8)

project(slow_macros)

macro(len_macro moc_flags)
  set (_moc_parameters ${moc_flags} -o "foo")
  message(STATUS "macro about to count length" )
  list(LENGTH _moc_parameters len)
  message(STATUS "macro produced length of: ${len}" )
endmacro()

macro(load_includes_macro include_var)
  include(LongIncludes.cmake)
  set(${include_var} ${short_includes})
  message(STATUS "Brought in long includes")
endmacro()

macro(fowarding_scope_macro )
  load_includes_macro(moc_flags)
  message(STATUS "fowarding_scope_macro invoked")
  len_macro("${moc_flags}")
endmacro()

function(len_function moc_flags)
  set (_moc_parameters ${moc_flags} -o "foo")
  message(STATUS "function about to count length" )
  list(LENGTH _moc_parameters len)
  message(STATUS "function produced length of: ${len}" )
endfunction()

function(load_includes_function include_var)
  include(LongIncludes.cmake)
  set(${include_var} ${short_includes} PARENT_SCOPE)
  message(STATUS "Brought in long includes")
endfunction()

function(fowarding_scope_function )
  load_includes_function(moc_flags)
  message(STATUS "fowarding_scope_function invoked")
  len_function( "${moc_flags}" )
endfunction()


fowarding_scope_function()

fowarding_scope_macro()

