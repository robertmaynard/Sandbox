Develop your Castle making skills in the Sandbox.


## Check CXX Symbols Exist ##
A look at what we would have to change to have CMake find templated
function signatures.


## Cuda Ptxas ##
Trying to replicate a cuda compile bug I encountered when building dax, with
Cuda 5.5


## Conceptual Executive ##
An attempt at a delayed worklet invocation pipeline for dax.


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


## FindTBB ##
A simplified FindTBB module. Going to be proposed to be part of CMake modules.


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


## Low Level Dax ##
An example of writing vis algorithms using everything but the Dax Scheduler
infrastructure.


## MoabReader ##
A pretty feature complete VTK reader of MOAB files, can output a moab
file a single unstructured grid, poly data or multi block.


## NinjaCustomCommand ##
A test case that shows a bug with ninja and CMake custom command.


## PortConcept ##
More brainstorming about filters and worklets in DAX.


## ShibokenTest ##
Trying to replicate some bugs I have encountered with Shiboken


## TryCompile ##
A basic example of doing a try compile in CMake. In This example we are
looking at if we have shared_ptr support
