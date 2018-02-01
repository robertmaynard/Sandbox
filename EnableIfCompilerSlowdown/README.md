Reproduction of the clang compiler issue #36158


https://twitter.com/ericniebler/status/958490446107361280?refsrc=email&s=11&ref_src=twcamp%5Eshare%7Ctwsrc%5Eios%7Ctwgr%5Eemail
https://bugs.llvm.org/show_bug.cgi?id=36158
https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60976#c19


On my machine, compiling with SFINAE_FAST defined takes 0.58s, where as without the define, it takes 5.8s.

By comparison, gcc is also slower for SFINAE-in-template-parameter-list, but "only" about 60%.

CONTEXT:
The popular range-v3 library makes extensive use of the SFINAE-in-template-parameter-list to emulate concepts. This perf bug is probably severely effecting compile times.
