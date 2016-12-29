Develop your Castle making skills in the Sandbox.


## AllocTest ##
Benchmark how much it costs to use multiple allocations over a single
large allocation


## BranchPredictor ##
A simple program that shows the strength of the modern branch predictor, and
why you should make your data branch is more predicatable ways.

Note this was a rip from the SO question: http://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster-than-an-unsorted-array?rq=1


## Check CXX Symbols Exist ##
A look at what we would have to change to have CMake find templated
function signatures.


## CMakeCrossCompiler ##
Show how to use a CMake ToolChain file as the CMAKE_USER_MAKE_RULES_OVERRIDE.
This is useful if you have required compiler flags you need set for
both verifying the compiler works and for making all your targets.

This is really an exercise in understanding the lack of interaction between
toolchains / cross compilation and CMAKE_USER_MAKE_RULES_OVERRIDE


## CMakeImportedLibrary ##
A basic example at showing how to create an imported library in CMake, this
needs to be updated to show off the new import library features


## CMakePreferStaticLibs ##
Show how you can force CMake to prefer static libraries over dynamic
libraries.


## CMakeSlowEval ##
Example that shows how very long strings passed a parameter to a CMake macro
can cause a massive slowdown in performance.


## CMakeSystemConfigs ##
Show how to setup and use system specific configuration files.


## CompileTimeCrossProduct ##
An example of a compile time cross product generator for 2 variadic parameter
packs.

## CompileTimeIntersection ##
An example of a compile time intersection generator for 2 variadic parameter
packs.


## CudaStackCopy ##
Checking the limits of what you can construct on the stack in CUDA.


## DumpTags ##
Code to dump information about moab files. Also provides SimpleMoab.h
which is a more c++ like interface to the majority of moab functions.


## ExtendTypedef ##
Meta template utilities to modify and extend a typedef that represents
a function call. For example it can take something like:
```
void Sig(_1,_2,_3)
```

and convert it to
```
void Sig(_1,_2,_4,_3)
```

This is all research code to be used in DAX, and this should be only
used as a draft example of the final implementation.


## ExternC ##
An example of writing an external C function when using CMake


## FindMoab ##
A proof of concept FindMOAB.cmake implementation to be used as a reference.
Also provided is the basic layout of a MOABConfig.cmake, that moab could
write out to make finding moab not require a FindMOAB.cmake

Also we show how to use FindMoab and verify it works by building a basic
example

## FindBoostHeaders ##
A version of FindBoost that only has the logic to find the header components
of boost. This is really handy when a project doesn't care about the boost
libraries.


## FizzBuzz ##
A template implementation of [FizzBuzz](http://www.codinghorror.com/blog/2007/02/why-cant-programmers-program.html) that should blow your mind ( or stack )


## FunctorTrampoline ##
An of how to wrap and pass an arbitrary  functor to an function so that
we don't leak OpenMP pragmas in a header.


## IdListToString ##
Example of converting vtkIdLists or vtkIdType* to a string key that can be used
for unique comparison. This string that is generated will be based on the sorted
order of the ids


## IntegerSequence ##
A simple example to show how to do compile time integer sequence creation.
This version does 8 item blocks, so you can go above 1024 integer values.


## MinDelta ##
A C++ example of making sure that two double values have at-least 64k
representable numbers between them.


## MoabReader ##
A pretty feature complete VTK reader of MOAB files, can output a moab
file a single unstructured grid, poly data or multi block.


## MSVCParserErrors ##
A collection of examples of how to compile the MSVC C++ parser

The simpliest version is now:
```
struct F {};
typedef void example(F(F,F), F(F,F,F));
```


## NinjaCustomCommand ##
A test case that shows a bug with ninja and CMake custom command.


## NumericLimits ##
Compare the VTK numeric type limits to the ones provided in C++11.


## OSXDylibLimits ##
A simple example that exposes a bug with @executable_path and OSX 10.8+.

It seems that the dylib loader has a fixed size memory pool for expanding
@executable_path, and either resolving that to a extremely long path, or
instead resolving hundreds of libraries that uses @executable_path cause
this to fail.

Currently the easiest way to reproduce this problem is by using the paraview
binary as it has a sufficient number of libraries to expose the problem
when placed in a directory path that contains ~130 characters.


## PointMergingBenchmark ##
Showing the most efficient way to merge duplicate point ids, will need
to be ported to VTK.


## ReduceTextualSymbols ##
Exploration on how best to reduce the number of internal textual symbols
are generated when building a library that is composed of numerous templated functions and methods.


## VtkComputeRangeBenchmark ##
Benchmarking the old compute range code in VTK compared to the new proposed
version, to show that the new version is faster.
Edit: The new version of compute range has been merged.

## VtkmVectoriziable ##
Very small snippets of code to verify the vtkm::Vec generates vectorized
SIMD code.
