//Requires C++11 support enabled

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

#define VTK_FLOAT_MIN               VTK_TYPE_CAST(float, -1.0e+38f)
#define VTK_FLOAT_MAX               VTK_TYPE_CAST(float,  1.0e+38f)

#define VTK_DOUBLE_MIN              VTK_TYPE_CAST(double, -1.0e+299)
#define VTK_DOUBLE_MAX              VTK_TYPE_CAST(double,  1.0e+299)

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

template<typename T>
void show_limits(T vtkLowestValue, T vtkLargestValue,
                 T stlLowestValue, T stlLargestValue )
{
  std::cout << "Min Limits are the same " << (( vtkLowestValue == stlLowestValue ) ? "Yes" : "No") << std::endl;
  std::cout << "Max Limits are the same " << (( vtkLowestValue == stlLowestValue ) ? "Yes" : "No") << std::endl;

  std::cout << "VTK Limits" << std::endl;
  std::cout << "lowest: " << vtkLowestValue << std::endl;
  std::cout << "max:    " << vtkLargestValue << std::endl;

  std::cout << "STD Limits" << std::endl;
  std::cout << "lowest: " << stlLowestValue << std::endl;
  std::cout << "max:    " << stlLargestValue << std::endl;

  std::cout << "VTK Hex Representation" << std::endl;
  std::cout << "l:      " << hex_representation(vtkLowestValue) << std::endl;
  std::cout << "m:      " << hex_representation(vtkLargestValue) << std::endl;

  std::cout << "STD Hex Representation" << std::endl;
  std::cout << "l:      " << hex_representation(stlLowestValue) << std::endl;
  std::cout << "m:      " << hex_representation(stlLargestValue) << std::endl;

  std::cout << std::endl;
}


int main(int argc, char const *argv[])
{
std::cout << "Long Limits " << std::endl;
show_limits<long>(VTK_LONG_MIN,
                  VTK_LONG_MAX,
                  std::numeric_limits<long>::lowest(),
                  std::numeric_limits<long>::max());

std::cout << "Float Limits " << std::endl;
show_limits<float>(VTK_FLOAT_MIN,
                  VTK_FLOAT_MAX,
                  std::numeric_limits<float>::lowest(),
                  std::numeric_limits<float>::max());

std::cout << "Double Limits " << std::endl;
show_limits<double>(VTK_DOUBLE_MIN,
                  VTK_DOUBLE_MAX,
                  std::numeric_limits<double>::lowest(),
                  std::numeric_limits<double>::max());

  return 0;
}
