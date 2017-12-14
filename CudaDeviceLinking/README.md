An example of how CMake should form the device linking command line

The existing issue with how CMake does device linking is that it drops all
shared libraries and unknown libraries from the command line.

This is problematic as these unknown libraries can have device symbols that
you want resolved.

Question 1:

I had originally wondered if the way we pass things to the nvcc -dlink made
a difference ('-lcublas_device' versus 'cublas_device.a') but in testing
both of these signatures cause the device symbols from cublas_device to be baked
into the output object file:

```
duplicate symbol ___cudaRegisterLinkedBinary_51_tmpxft_0000561c_00000000_15_axpy_compute_60_cpp1_ii_c7deafe8 in:
    /Users/robert/Work/Sandbox/src/CudaDeviceLinking/build/libstatic_1.a(cmake_device_link_1.o)
    /Users/robert/Work/Sandbox/src/CudaDeviceLinking/build/libstatic_2.a(cmake_device_link_2.o)
```


Question 2:

Should we be device linking shared libraries, or only static libraries???
