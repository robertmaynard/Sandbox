#ifdef _WIN32
#  define EXPORT __declspec(dllexport)
#else
#  define EXPORT
#endif

EXPORT bool lib_kernel(int x, int& r)
{
  r = -x;
  return true;
}
