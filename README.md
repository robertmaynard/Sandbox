Develop your Castle making skills in the Sandbox.


## AllocTest ##
Benchmark how much it costs to use multiple allocations over a single
large allocation


## AtC ##
Efficient implementation of at_c for extracting a type from a type list, which
is useful when doing C++11 Meta Template Programming.


## BranchPredictor ##
A simple program that shows the strength of the modern branch predictor, and
why you should make your data branch is more predicatable ways.

Note this was a rip from the SO question: http://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster-than-an-unsorted-array?rq=1


## BytesToHumanReadable ##
Multiple efficient implementation of converting a `std::uint64_t` 
that represents bytes into a human readable string like `256.23GB`.

Requires google benchmark


## Check CXX Symbols Exist ##
A look at what we would have to change to have CMake find templated
function signatures.


## CMakeCrossCompiler ##
Show how to use a CMake ToolChain file as the CMAKE_USER_MAKE_RULES_OVERRIDE.
This is useful if you have required compiler flags you need set for
both verifying the compiler works and for making all your targets.

This is really an exercise in understanding the lack of interaction between
toolchains / cross compilation and CMAKE_USER_MAKE_RULES_OVERRIDE


## CMakeCustomCommandAfterTarget ##
Show how to make sure a custom command comes between
two targets


## CMakeCustomPlatform ##
An attempt at making a custom platform for cmake that requires weird extra
flags to compile/link. FIY: the best way to do this is to add the compiler 
directly to CMake.


## CMakeDebugLinkFiles ##
Show how to use write custom CMake code to allow debug information to be removed
from a library at install time and placed into a separate file


## CMakeEnableC99Manually ##
Show how to enable C99 and C++11 on files in a target if you can't require
CMake 3.1 or higher


## CMakeForceCUDAIncludeAsSystem ##
Show how to force an implicit CUDA system include path to be `isystem` for
the CXX compiler.


## CMakeImportedLibrary ##
A basic example at showing how to create an imported library in CMake, this
needs to be updated to show off the new import library features


## CMakeInterfaceChangesLibrary ##
A basic example at showing how you can have an interface library provide
different libraries to link too based on a configuration property of the
target that is CONSUMING the interface library.


## CMakePreferStaticLibs ##
Show how you can force CMake to prefer static libraries over dynamic
libraries.


## CMakeSlowEval ##
Example that shows how very long strings passed a parameter to a CMake macro
can cause a massive slowdown in performance.


## CMakeSystemConfigs ##
Show how to setup and use system specific configuration files.


## CTestTimingLogic ##
Reverse how CTest tracks the valid timing window for a test


## CompileTimeCrossProduct ##
An example of a compile time cross product generator for 2 variadic parameter
packs.


## CompileTimeIntersect ##
An example of a compile time intersection generator for 2 variadic parameter
packs.


## CudaDeviceLinking ##
Exploration on how to best form the link line for nvcc device linking


## CudaStackCopy ##
Checking the limits of what you can construct on the stack in CUDA.


## CudaThreadsPerBlock ##
Toy code that tunes the number of threads per block using compile time
heuristics.


## DumpTags ##
Code to dump information about moab files. Also provides SimpleMoab.h
which is a more c++ like interface to the majority of moab functions.


## EnableIfCompilerSlowdown ##
Reproduction of the clang compiler issue #36158. Basically a quick way
to show how enable_if can effect compile time performance


## ExplicitTemplate ##
An example of forcing explicit template instantiation in a translation unit.
Shows how implicit and explicit effect symbol visibility of functions.


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


## FindBoostHeaders ##
A version of FindBoost that only has the logic to find the header components
of boost. This is really handy when a project doesn't care about the boost
libraries.


## FindMoab ##
A proof of concept FindMOAB.cmake implementation to be used as a reference.
Also provided is the basic layout of a MOABConfig.cmake, that moab could
write out to make finding moab not require a FindMOAB.cmake

Also we show how to use FindMoab and verify it works by building a basic
example


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

The simplest version is now:
```
struct F {};
typedef void example(F(F,F), F(F,F,F));
```

See: https://github.com/robertmaynard/compiler_tests for an up to
date version of this and other compiler regressions.


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
when placed in a directory path that contains \~130 characters.


## OSXRttiIssue ##
Show how setting the default visibility as hidden and than having a base
class as public can cause issues when going across library boundaries


## PointMergingBenchmark ##
A simple program that compares different way to resolve duplicate points.
We compare vtk's vtkMergePoints, vtkIncrementalOctreePointLocator, and
using a std::vector that is sorted, uniqueified and than scan with lower_bounds
to find the new values.

Not unsurprising to me, using a std::vector and the sort, unique, lower_bounds
is the best for performance.

## ReduceTextualSymbols ##
Exploration on how best to reduce the number of internal textual symbols
are generated when building a library that is composed of numerous templated functions and methods.
Tips to keep a reduced number of textual symbols:
```
1. Use the unnamed namespaces
2. Use static inline functions
3. Use force_inline compiler directives
```

## TypedefConverter ##
A python script that converts typedef over the new C++11 using syntax. Was 
developed as clang-tidy incorrectly replaces typedef such as:
```cpp
typedef typename V::T T;
//clang-tidy would replace convert this too the following when V::T is a float
using T = float;
//When we really want the code to look like
using T = typename V::T;
```


## VtkComputeRangeBenchmark ##
Benchmarking the old compute range code in VTK compared to the new proposed
version, to show that the new version is faster.
Edit: The new version of compute range has been merged.


## vtkmEasyExecutionType ##
Very small snippets of code to verify how to make getting the portal
types from an ArrayHandle for a given device easier.


## vtkmBenchmarkUploader ##
A collection of python programs that can run benchmark executables saving the
results to 'versioned' files which than can be uploaded directly to s3 or first
converted into json and than uploaded to s3

See the subdirectory Readme for more information on how to use the collection
of tools


## VtkmVectoriziable ##
Very small snippets of code to verify the vtkm::Vec generates vectorized
SIMD code.

