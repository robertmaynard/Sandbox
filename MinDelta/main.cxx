

#include <iostream>
#include <iomanip>
#include <cstring>

//----------------------------------------------------------------------------
template<typename T, typename EquivSizeIntT>
bool AdjustTRange(T range[2], EquivSizeIntT)
{
  if (range[1] < range[0])
    {
    // invalid range.
    return false;
    }

  const bool spans_zero_boundary = range[0] < 0 && range[1] > 0;
  if(spans_zero_boundary)
    { //nothing needs to be done, but this check is required.
      //if we convert into integer space the delta difference will overflow
      //an integer
    return false;
    }

  EquivSizeIntT irange[2];
  //needs to be a memcpy to avoid strict aliasing issues
  std::memcpy(irange, range, sizeof(T)*2);

  const EquivSizeIntT minDelta = 65536;

  //determine the absolute delta between these two numbers.
  EquivSizeIntT delta = std::abs(irange[1] - irange[0]);

  //if our delta is smaller than the min delta push out the max value
  //so that it is equal to minRange + minDelta. When our range is entirely
  //negative we should instead subtract from our max, to max a larger negative
  //value
  if(delta < minDelta && irange[1] < 0)
    {
    irange[1] = irange[0] - minDelta;
    //needs to be a memcpy to avoid strict aliasing issues
    std::memcpy(range+1, irange+1, sizeof(T) );
    return true;
    }
  if(delta < minDelta)
    {
    irange[1] = irange[0] + minDelta;
    //needs to be a memcpy to avoid strict aliasing issues
    std::memcpy(range+1, irange+1, sizeof(T) );
    return true;
    }
  return false;
}

//----------------------------------------------------------------------------
bool AdjustRange(float range[2])
{
  return AdjustTRange(range, int());
}

//----------------------------------------------------------------------------
bool AdjustRange(double range[2])
{
  if( range[0] > -2 && range[0] < 2 )
    {
    float frange[2] = { static_cast<float>(range[0]), static_cast<float>(range[1]) };
    bool result = AdjustTRange( frange, int() );
    if(result)
      {
      range[0] = static_cast<double>(frange[0]);
      range[1] = static_cast<double>(frange[1]);
      }
    return result;
    }
  long long intType = 0;
  return AdjustTRange(range, intType);

}

//----------------------------------------------------------------------------
void test_range(double range[2] )
{
  std::cout << "------------------------------------------------------------" << std::endl;
  std::cout << "input range: " << range[0] << " : " << range[1] << std::endl;
  AdjustRange(range);
  std::cout << "output range: " << std::fixed << std::setprecision(16) << range[0] << " : " << range[1] << std::endl;
  std::cout << std::endl;
}


int main(int, char **)
{
  double zeros[2] = { 0.0, 0.0 };
  double ones[2] = { 1.0, 1.0 };
  double nones[2] = { -1.0, -1.0 };
  double zero_one[2] = { 0.0, 1.0 };
  double none_one[2] = { -1.0, 1.0 };
  double none_zero[2] = { -1.0, 1.0 };

  double small[2] = { -12, -4};
  double large[2] = { 1e12, 1e12+1 };
  double real_small[2] = { 1e-20, 1e-19 };

  test_range(zeros);
  test_range(ones);
  test_range(nones);
  test_range(zero_one);
  test_range(none_one);
  test_range(none_zero);

  test_range(small);
  test_range(large);
  test_range(real_small);

  return 0;
}
