
#include <vtkNew.h>
#include "convert.h"

#include <sys/time.h>

#include <algorithm>
#include <numeric>

void random_fill(vtkIdList* list, int max_count, int max_value)
{
  //srand on time
  srand (time(NULL));

  //start random seed
  timeval currentTime;
  gettimeofday(&currentTime, NULL);

  vtkIdType size = 1 + ((int)currentTime.tv_usec % (max_count-1) );
  list->SetNumberOfIds(size);

  for(int i=0; i < size; ++i)
  {
    list->SetId(i,(int)currentTime.tv_usec % max_value);
  }

}

void reverseList(vtkIdList* in, vtkIdList* out)
{
  out->DeepCopy(in);
  vtkIdType size = out->GetNumberOfIds();
  std::reverse(out->GetPointer(0),out->GetPointer(size));
}
int main(int, char**)
{
  //we will test 1000 random list of values each with any where from
  //1 to 10,000 values. the generated values will be between 0 and 1million
  for(int i=0; i < 1000; ++i)
  {
    vtkNew<vtkIdList> listToTest;
    //10000 max value size, 1mill max id value
    random_fill(listToTest.GetPointer(),10000,1000000);

    std::string result = to_key(listToTest.GetPointer());

    if(result != to_key(listToTest->GetPointer(0),listToTest->GetNumberOfIds()) )
    {
    std::cerr << "key's failed to match when past by pointer"  << std::endl;
    return 1;
    }

    vtkNew<vtkIdList> reversedList;
    reverseList(listToTest.GetPointer(),reversedList.GetPointer());
    std::string reversedResult = to_key(reversedList.GetPointer());

    if(result != reversedResult )
    {
    std::cerr << "key's failed to match given reversed order" << std::endl;
    return 1;
    }

    std::cout << "gen key: " << result << " len: " <<  listToTest->GetNumberOfIds() << std::endl;
  }

return 0;
}

