#include <algorithm>
#include <ctime>
#include <iostream>


long long sum_array(int* data, unsigned int size)
{
  long long sum=0;
  for (unsigned i = 0; i < 100000; ++i)
  {
  // Primary loop
  for (unsigned c = 0; c < size; ++c)
    {
      if (data[c] >= 128)
        sum += data[c];
    }
  }
  return sum;
}

int main()
{
    //measurement info
    clock_t start;
    long long sum;
    double elapsedTime;

    // Generate data
    const unsigned arraySize = 32768;
    int data[arraySize];

    for (unsigned c = 0; c < arraySize; ++c)
        data[c] = std::rand() % 256;


    // Test unsorted performance
    start = clock();
    sum = sum_array(data,arraySize);
    elapsedTime = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
    std::cout << "unsorted time: " << elapsedTime << std::endl;
    std::cout << "sum = " << sum << std::endl;

    //sort data and record time
    start = clock();
    std::sort(data, data + arraySize);
    elapsedTime = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
    std::cout << std::endl;
    std::cout << "sort time: " << elapsedTime << std::endl;

    // Test sorted performance
    start = clock();
    sum = sum_array(data,arraySize);
    elapsedTime = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
    std::cout << std::endl;
    std::cout << "sorted time: " << elapsedTime << std::endl;
    std::cout << "sum = " << sum << std::endl;
}