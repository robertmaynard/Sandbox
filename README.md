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


## CMakeSlowEval ##
Example that shows how very long strings passed a parameter to a CMake macro
can cause a massive slowdown in performance.


## CMakeTrackConfFiles ##
Working on a test to verify that CMake only tracks the proper configured files, and
that ones that don't exist after generation aren't tracked


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


## FunctorTrampoline ##
An of how to wrap and pass an arbitrary  functor to an function so that
we don't leak OpenMP pragmas in a header.


## FizzBuzz ##
A template implementation of [FizzBuzz](http://www.codinghorror.com/blog/2007/02/why-cant-programmers-program.html) that should blow your mind ( or stack )


## GenericVariadicParser ##
An in progress attempt at being able to strip and pack function arguments
into a storage mechanism so that we can extract them latter. The main feature
is that subclasses can state how many arguments they want not packed in the
opaque container.

So basically you have parser that takes 10 paramenters. Than
derived parser can state it wants only the first two. It than has
the ability to add parameters or remove the ones it explicitly asked for.

Example:
```
void operator(Functor f, T param1, O param2, RestOfParameters rp)
{
params::invoke(f,param1,myInsertedParam,param2,rp);
}
```


## IdListToString ##
Example of converting vtkIdLists or vtkIdType* to a string key that can be used
for unique comparison. This string that is generated will be based on the sorted
order of the ids


## LHMC ##
An example of looking at using Low High tables to speed up operations such
as Marching Cubes


## Low Level Dax ##
An example of writing vis algorithms using everything but the Dax Scheduler
infrastructure.


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
  struct
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


## PortConcept ##
More brainstorming about filters and worklets in DAX.


## TryCompile ##
A basic example of doing a try compile in CMake. In This example we are
looking at if we have shared_ptr support


## VtkComputeRangeBenchmark ##
Benchmarking the old compute range code in VTK compared to the new proposed
version, to show that the new version is faster.
Edit: The new version of compute range has been merged.

