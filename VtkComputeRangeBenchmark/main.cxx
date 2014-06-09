
#include "Timer.h"

#include <vtkNew.h>
#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <typeinfo>
#include <vector>


template<typename T>
class Generator
{
public:
  typedef typename std::vector< T >::const_iterator const_iterator;

  Generator(std::size_t num_values)
  {
   T lower_bound = (std::numeric_limits<T>::lowest()+20);
   T upper_bound = (std::numeric_limits<T>::max()-20);

  Values.reserve(num_values);

  std::uniform_real_distribution<T> unif(lower_bound,upper_bound);
  std::default_random_engine re;
  for(int i=0; i < num_values; ++i)
    { Values.push_back( unif(re) ); }
  }

  int size() const { return Values.size(); }

  const_iterator begin() const { return Values.begin(); }
  const_iterator end() const { return Values.end(); }

private:
  std::vector< T  > Values;
};


template<typename T, typename DataArrayType>
void single_comp(std::size_t numElements)
{
  Generator<T> generator(numElements);

  vtkNew<DataArrayType> array;
  array->SetNumberOfComponents(1);
  array->SetNumberOfValues( numElements );

  T* dest = reinterpret_cast<T*>(array->GetVoidPointer(0));
  std::copy(generator.begin(), generator.end(),dest);

  std::cout << "Computing the range for a single component array of type: "
            << typeid(T).name() << " with a length of " << numElements << std::endl;

  Timer timer;

  array->GetRange();

  std::cout << timer.GetElapsedTime() << " seconds " << std::endl;
}

template<typename T, typename DataArrayType>
void multi_comp(std::size_t numElements, std::size_t numComps)
{
  Generator<T> generator(numElements*numComps);

  vtkNew<DataArrayType> array;
  array->SetNumberOfComponents( numComps );
  array->SetNumberOfTuples( numElements );

  T* dest = reinterpret_cast<T*>(array->GetVoidPointer(0));
  std::copy(generator.begin(), generator.end(),dest);

  std::cout << "Computing the range for a multiple component array of type: "
            << typeid(T).name() << " with a length of " << numElements
            << " and with " << numComps << " components " <<  std::endl;

  Timer timer;

  for(int i=0; i < numComps; ++i)
    { array->GetRange(i); }

  std::cout << timer.GetElapsedTime() << " seconds " << std::endl;

  std::cout << "Computing the magnitude for a multiple component array of type: "
            << typeid(T).name() << " with a length of " << numElements
            << " and with " << numComps << " components " <<  std::endl;

  timer.Reset();
  array->GetRange(-1);
  std::cout << timer.GetElapsedTime() << " seconds " << std::endl;
}


int main(int argc, char **argv)
{
  const int number_of_elements = 10000000; //lets start with 10 million

  single_comp<float, vtkFloatArray>(number_of_elements);
  single_comp<double, vtkDoubleArray>(number_of_elements);
  single_comp<int, vtkIntArray>(number_of_elements);
  single_comp<vtkIdType, vtkIdTypeArray>(number_of_elements);

  multi_comp<float, vtkFloatArray>(number_of_elements, 3);
  multi_comp<double, vtkDoubleArray>(number_of_elements, 3);
  multi_comp<int, vtkIntArray>(number_of_elements, 3);
  multi_comp<vtkIdType, vtkIdTypeArray>(number_of_elements, 3);


  multi_comp<float, vtkFloatArray>(number_of_elements, 9);
  multi_comp<double, vtkDoubleArray>(number_of_elements, 9);
  multi_comp<int, vtkIntArray>(number_of_elements, 9);
  multi_comp<vtkIdType, vtkIdTypeArray>(number_of_elements, 9);

  return 0;
}
