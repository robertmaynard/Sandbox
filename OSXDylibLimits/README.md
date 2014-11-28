A simple example that exposes a bug with @executable_path and OSX 10.8+.

It seems that the dylib loader has a fixed size memory pool for expanding
@executable_path, and either resolving that to a extremely long path, or
instead resolving hundreds of libraries that uses @executable_path cause
this to fail.

Currently the easiest way to reproduce this problem is by using the paraview
binary as it has a sufficient number of libraries to expose the problem
when placed in a directory path that contains ~130 characters.