
#include <vtkNew.h>
#include "convert.h"

int main(int, char**)
{
  vtkNew<vtkIdList> descending;
  descending->SetNumberOfIds(3);
  descending->SetId(0,5);
  descending->SetId(1,4);
  descending->SetId(2,3);

  std::string desc = to_key(descending.GetPointer());

  if(desc != to_key(descending->GetPointer(0),3) )
    {
    std::cerr << "key's failed to match given same input sequence past by pointer"  << std::endl;
    return 1;
    }

  std::cout << "descending key is: " << desc << std::endl;

  vtkNew<vtkIdList> ascending;
  ascending->SetNumberOfIds(3);
  ascending->SetId(0,3);
  ascending->SetId(1,4);
  ascending->SetId(2,5);

  std::string asc = to_key(ascending.GetPointer());

  if(desc != asc )
    {
    std::cerr << "key's failed to match given ascending and descending order" << std::endl;
    return 1;
    }

  std::cout << "descending key is: " << asc << std::endl;

  return 0;
}
