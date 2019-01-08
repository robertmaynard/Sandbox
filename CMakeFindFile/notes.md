@akuskis 
So the answer is complicated, and  my understanding could be wrong on certain finer details. So please
don't use this against CMake when it doesn't behave exactly like I state below :). 

Now the key to understanding `find` queries is understanding how CMake constructs and manages the concept of search paths.
The default paths that CMake searches for all `find` calls are composed of two primary
components the first is the prefix which is what is controllable through things such as `CMAKE_PREFIX_PATH` and
`CMAKE_FIND_ROOT_PATH` and the suffix which is determined based on which `find` call is occurring. ( way more information on this is covered in find_path docs under `NO_DEFAULT_PATH`).

The short of is that for `find_path` CMake will have determined the prefixes to search are  `/usr/local/`, `/usr/`, and `/` 
( in reality this is only the first 3 of 8 for Linux ). suffixes are `include/`, and `include/<arch>` ( again a cross simplification ). 
These combine with the prefixes to generate the search list of:
```
/usr/local/include/
/usr/local/include/<arch>/
/usr/include/
/usr/include/<arch>/
/include/
/include/<arch>/
```

For `find_path` the primary arguments to control the paths are `HINTS`, `PATH` and `PATH_SUFFIXES` each one filling
a different role when combined with the default paths that CMake will search. 

Now `PATH_SUFFIXES` is the easiest to cover. It is extra relative subdirectories to add to search search path.
So for the above example the `PATH_SUFFIXES` of `my_lib` would result in the search paths of:
```
/usr/local/include/
/usr/local/include/<arch>/
/usr/local/include/my_lib/
/usr/local/include/<arch>/my_lib/
/usr/include/
/usr/include/<arch>/
/usr/include/my_lib/
/usr/include/<arch>/my_lib/
/include/
/include/<arch>/
/include/my_lib/
/include/<arch>/my_lib/

```

`PATHS` and `HINTS` both effectively add additional absolute search paths to the logic. The biggest
different is the priority they have compared to the intrinsic/computed paths that are used by CMake.
`HINTS` is parsed 4th in order, before using standard environment variables and platform paths that CMake has
computed based on the platform. This hould be used when you have previous system introspection information. 
`PATHS` is used last (6th) and are your wild speculative guesses when all else fails


Okay so `CMAKE_FIND_ROOT_PATH`. Now `CMAKE_FIND_ROOT_PATH` is a collection of paths that are added as prefixes to cmake. 
So if you have a `CMAKE_FIND_ROOT_PATH` of `/home/robert/pi`  you would get the following search paths:
```
/home/robert/pi/usr/local/include/
/home/robert/pi/usr/local/include/<arch>/
/home/robert/pi/usr/include/
/home/robert/pi/usr/include/<arch>/
/home/robert/pi/include/
/home/robert/pi/include/<arch>/
...
```


So how does this relate to your example?

`find_path(X_DIR x)` fails since x is in `/tmp/c/` and not a directory covered by suffix. 
`find_path(Y_DIR test/y)` fails for same reasons. Something like `find_path(Y_DIR y PATH_SUFFIXES ../test/)` would work
`find_path(Z_DIR z)` works since z is in `include`
`find_path(W_DIR include/w)` doesn't work since `w` is in `include` and not `include/include/`
