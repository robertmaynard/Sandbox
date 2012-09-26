#include "vtkMoabReader.h"

#include "SimpleMoab.h"

#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMultiBlockDataSet.h"

vtkStandardNewMacro(vtkMoabReader)
//------------------------------------------------------------------------------
vtkMoabReader::vtkMoabReader()
  {
  }

//------------------------------------------------------------------------------
vtkMoabReader::~vtkMoabReader()
  {
  }

//------------------------------------------------------------------------------
int vtkMoabReader::RequestInformation(vtkInformation *vtkNotUsed(request),
                       vtkInformationVector **vtkNotUsed(inputVector),
                       vtkInformationVector *vtkNotUsed(outputVector))
{

  //todo. Walk the file and display all the 2d and 3d elements that the users
  //could possibly want to load
  return 1;
}

//------------------------------------------------------------------------------
int vtkMoabReader::RequestData(vtkInformation *vtkNotUsed(request),
                vtkInformationVector **vtkNotUsed(inputVector),
                vtkInformationVector *vtkNotUsed(outputVector))
{
  //walk the subset the user selected and read only those in
  return 1;
}

//------------------------------------------------------------------------------
void vtkMoabReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

