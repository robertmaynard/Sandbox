

#include <algorithm>
#include <string>
#include <vector>
#include <vtkIdList.h>
#include <vtksys/MD5.h>


std::string to_key(const vtkIdType* data, vtkIdType len)
{
  //sort
  std::vector<vtkIdType> sorted(data,data+len);
  std::sort(sorted.begin(),sorted.end());

  //hash the raw memory of the vector as a string we have to give it
  //a length or else the method will try to do strlen which will be very bad.
  //md5 really doesn't care if what type the data is anyhow
  vtksysMD5* hasher = vtksysMD5_New();
  vtksysMD5_Initialize(hasher);

  //sizeof returns the number of character per vtkIdType and mult be len
  const std::size_t charLen = sizeof(vtkIdType) * len;

  unsigned const char* cdata = reinterpret_cast<unsigned const char*>(&sorted[0]);
  vtksysMD5_Append(hasher, cdata, charLen);

  //convert the hash to a string a return
  char hash[33];
  vtksysMD5_FinalizeHex(hasher, hash);
  vtksysMD5_Delete(hasher);
  hash[32] = 0;
  return std::string(hash);
}

std::string to_key(vtkIdList* list)
{
  return to_key(list->GetPointer(0),list->GetNumberOfIds());
}
