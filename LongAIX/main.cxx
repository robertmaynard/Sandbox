#include <cmath>
#include <limits>
#include <iostream>
#include <sstream>


#if defined(__cplusplus)
# define VTK_TYPE_CAST(T, V) static_cast< T >(V)
#else
# define VTK_TYPE_CAST(T, V) ((T)(V))
#endif

#define VTK_LONG_MIN                VTK_TYPE_CAST(long, ~(~0ul >> 1))
#define VTK_LONG_MAX                VTK_TYPE_CAST(long, ~0ul >> 1)
#define STD_LONG_MIN                VTK_TYPE_CAST(long, std::numeric_limits<long>::min())
#define STD_LONG_MAX                VTK_TYPE_CAST(long, std::numeric_limits<long>::max())

template<typename T>
std::string hex_representation(T t1)
{
  static char const hex[16] =
    {'0', '1', '2', '3', '4', '5', '6', '7',
     '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

  unsigned char* input = reinterpret_cast<unsigned char*>(&t1);
  std::string output;
  for(std::size_t i=0; i < sizeof(T); ++i)
    {
    output += hex[input[i] >> 4];
    output += hex[input[i] & 0xF];
    }
  return output;
}

int main(int argc, char const *argv[])
{
  long vlmin(VTK_LONG_MIN);
  long vlmax(VTK_LONG_MAX);

  long slmin(STD_LONG_MIN);
  long slmax(STD_LONG_MAX);

  std::cout << "Min Limits are the same " << ( vlmin == slmin ) << std::endl;
  std::cout << "Max Limits are the same " << ( vlmax == slmax ) << std::endl;


  std::cout << "VTK Limits" << std::endl;
  std::cout << vlmin << std::endl;
  std::cout << vlmax << std::endl;

  std::cout << "STD Limits" << std::endl;
  std::cout << slmin << std::endl;
  std::cout << slmax << std::endl;

  std::cout << "VTK Hex Representation" << std::endl;
  std::cout << hex_representation(vlmin) << std::endl;
  std::cout << hex_representation(vlmax) << std::endl;
  std::cout << std::hex << vlmin << std::endl;
  std::cout << std::hex << vlmax << std::endl;

  std::cout << "STD Hex Representation" << std::endl;
  std::cout << hex_representation(slmin) << std::endl;
  std::cout << hex_representation(slmax) << std::endl;
  std::cout << std::hex << slmin << std::endl;
  std::cout << std::hex << slmax << std::endl;
  return 0;
}