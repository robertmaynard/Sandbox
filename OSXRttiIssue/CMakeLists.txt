project(rtti)

add_library(RangeHidden SHARED range.cpp)
add_library(RangeMixed SHARED range.cpp)
add_library(RangePublic STATIC range.cpp)

target_compile_features(RangeHidden PUBLIC cxx_constexpr)
target_compile_features(RangeMixed PUBLIC cxx_constexpr)
target_compile_features(RangePublic PUBLIC cxx_constexpr)

add_executable(Hidden consume.cpp)
target_link_libraries(Hidden PRIVATE RangeHidden)

add_executable(Mixed consume.cpp)
target_link_libraries(Mixed PRIVATE RangeMixed)

add_executable(Visible consume.cpp)
target_link_libraries(Visible PRIVATE RangePublic)

#Setting hidden vis on produce or consume breaks everything
target_compile_options(RangeHidden PUBLIC -fvisibility=hidden)
target_compile_options(RangeMixed PUBLIC -fvisibility=hidden)
target_compile_options(RangePublic PUBLIC -fvisibility=default)

target_compile_definitions(RangeMixed PUBLIC -DMIX_VIS)

