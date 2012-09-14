
#this function will define a parent scope
#variable with the name of variable you pass in
#if you need to find boost it will also define
#a variable named ${type}_BOOST_TRUE

function(determineSharedPtrType type incType)

  set(BOOST_TYPE 3)
  set(RESULT 0)
  set(SHARED_PTR_TYPE_FOUND FALSE)

  if(NOT ${SHARED_PTR_TYPE_FOUND})
    try_compile(SHARED_PTR_TYPE_FOUND
      ${PROJECT_BINARY_DIR}/CMakeTmp
      ${PROJECT_SOURCE_DIR}/CMake/shared_ptr.cxx
      )
    if(${SHARED_PTR_TYPE_FOUND})
      set(RESULT "std::shared_ptr")
      set(INCLUDE_RESULT "memory")
    endif()
  endif()

  if(NOT ${SHARED_PTR_TYPE_FOUND})
  try_compile(SHARED_PTR_TYPE_FOUND
    ${PROJECT_BINARY_DIR}/CMakeTmp
    ${PROJECT_SOURCE_DIR}/CMake/shared_ptr_tr1.cxx
    )
    if(${SHARED_PTR_TYPE_FOUND})
      set(RESULT "std::tr1::shared_ptr")
      set(INCLUDE_RESULT "tr1/memory")
    endif()
  endif()

  if(NOT ${SHARED_PTR_TYPE_FOUND})
    set(RESULT "boost::shared_ptr")
    set(INCLUDE_RESULT "boost/shared_ptr.hpp")
    set(${type}_BOOST_TRUE TRUE PARENT_SCOPE)
  endif()


  set(${type} ${RESULT} PARENT_SCOPE)
  set(${incType} ${INCLUDE_RESULT} PARENT_SCOPE)

endfunction(determineSharedPtrType)

