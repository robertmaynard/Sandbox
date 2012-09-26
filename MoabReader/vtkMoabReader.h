#ifndef VTKMOABREADER_H
#define VTKMOABREADER_H

#include "vtkIOGeometryModule.h" // For export macro
#include "vtkMultiBlockDataSetAlgorithm.h"

class vtkInformation;
class vtkInformationVector;

class vtkMoabReader : public vtkMultiBlockDataSetAlogrithm
{
public:
  static vtkMOABReader *New();
  vtkTypeMacro(vtkMOABReader,vtkMultiBlockDataSetAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Specify file name of the MOAB mesh file.
  vtkSetStringMacro(FileName);
  vtkGetStringMacro(FileName);

protected:
  vtkMoabReader();
  ~vtkMoabReader();

  int RequestInformation(vtkInformation *vtkNotUsed(request),
                         vtkInformationVector **vtkNotUsed(inputVector),
                         vtkInformationVector *outputVector);

  int RequestData(vtkInformation *vtkNotUsed(request),
                  vtkInformationVector **vtkNotUsed(inputVector),
                  vtkInformationVector *outputVector);
private:
  vtkMOABReader(const vtkMOABReader&);  // Not implemented.
  void operator=(const vtkMOABReader&);  // Not implemented.
  char* FileName;

};

#endif // VTKMOABREADER_H
