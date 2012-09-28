#include "vtkMoabReader.h"

#include "SimpleMoab.h"
#include "DataSetConverter.h"

#include "vtkNew.h"
#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkUnstructuredGrid.h"


vtkStandardNewMacro(vtkMoabReader)
//------------------------------------------------------------------------------
vtkMoabReader::vtkMoabReader()
  {
  this->SetNumberOfInputPorts(0);
  this->FileName = NULL;
  }

//------------------------------------------------------------------------------
vtkMoabReader::~vtkMoabReader()
  {
  }

//------------------------------------------------------------------------------
int vtkMoabReader::RequestInformation(vtkInformation *request,
                       vtkInformationVector **inputVector,
                       vtkInformationVector *outputVector)
{

  //todo. Walk the file and display all the 2d and 3d elements that the users
  //could possibly want to load
  return this->Superclass::RequestInformation(request,inputVector,outputVector);
}

//------------------------------------------------------------------------------
int vtkMoabReader::RequestData(vtkInformation *vtkNotUsed(request),
                vtkInformationVector **vtkNotUsed(inputVector),
                vtkInformationVector *outputVector)
{
  //First pass is lets load in all 3d elements in a block called Volumes,
  //and load all 2d elements in a block called Surfaces

  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  vtkMultiBlockDataSet *output =
    vtkMultiBlockDataSet::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

  vtkNew<vtkMultiBlockDataSet> volumeRoot;
  vtkNew<vtkMultiBlockDataSet> surfaceRoot;

  const int blockIndex = output->GetNumberOfBlocks();
  output->SetBlock(blockIndex,volumeRoot.GetPointer());
  output->SetBlock(blockIndex+1,surfaceRoot.GetPointer());

  //boring work, set the names of the blocks
  output->GetMetaData(blockIndex)->Set(vtkCompositeDataSet::NAME(), "Volumes");
  output->GetMetaData(blockIndex+1)->Set(vtkCompositeDataSet::NAME(), "Surfaces");

  this->CreateSubBlocks(volumeRoot, 3);
  this->CreateSubBlocks(surfaceRoot, 2);

  return 1;
}


//------------------------------------------------------------------------------
void vtkMoabReader::CreateSubBlocks(vtkNew<vtkMultiBlockDataSet> & root,
                                  int dimensionality)
{
  //basic premise: query the database for all 3d elements and create a new
  //multiblock elemenent for each

  smoab::Interface interface(this->FileName);
  smoab::DataSetConverter converter(interface);

  smoab::EntityHandle rootHandle = interface.getRoot();
  smoab::Range parents = interface.findEntityRootParents(rootHandle);
  smoab::Range dimEnts = interface.findEntitiesWithTag(
                           smoab::GeomTag(dimensionality),
                           rootHandle);

  smoab::Range geomParents = smoab::intersect(parents,dimEnts);

  parents.clear(); //remove this range as it is unneeded
  dimEnts.clear();

  //now each item in range can be extracted into a different grid
  typedef smoab::Range::iterator iterator;
  vtkIdType index = 0;
  for(iterator i=geomParents.begin(); i != geomParents.end(); ++i, ++index)
    {
    vtkNew<vtkUnstructuredGrid> block;
    root->SetBlock(index,block.GetPointer());

    converter.fill(*i, block.GetPointer());
    }
}

//------------------------------------------------------------------------------
void vtkMoabReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

