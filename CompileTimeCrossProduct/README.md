A collection of examples to show how to do compile time cross product of two
parameter packs, using c++11

The old style causes the Intel compiler to hang forever when one of the lists
has more than ten items

The lazy style causes MSVC 2013 to error out, but works under Intel

The non lazy style works with MSVC 2013, but generally takes more system memory
than the lazy style

The lazy and non lazy style come directly from brigands tests
