A simple pyhton script that allows conversion of C++ typedef to C++11 using syntax

This is done as refactoring tools such as clang-tidy `modernize-use-using`
incorrectly replace the typedef with the first instantiation of the typedef,
instead of the more general template type.

We also have an example header that can be used to see how the conversion
works
