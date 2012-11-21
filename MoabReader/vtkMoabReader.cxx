#include "vtkMoabReader.h"

#include "SimpleMoab.h"
#include "ExtractShell.h"
#include "DataSetConverter.h"


#include "vtkNew.h"
#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkUnstructuredGrid.h"

#include "vtkPolyData.h"


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
  vtkNew<vtkMultiBlockDataSet> boundaryRoot;
  vtkNew<vtkMultiBlockDataSet> surfaceRoot;
  vtkNew<vtkMultiBlockDataSet> neumannRoot;
  vtkNew<vtkMultiBlockDataSet> dirichletRoot;

  const int blockIndex = output->GetNumberOfBlocks();
  output->SetBlock(blockIndex,volumeRoot.GetPointer());
  output->SetBlock(blockIndex+1,boundaryRoot.GetPointer());
  output->SetBlock(blockIndex+2,surfaceRoot.GetPointer());
  output->SetBlock(blockIndex+3,neumannRoot.GetPointer());
  output->SetBlock(blockIndex+4,dirichletRoot.GetPointer());

  //boring work, set the names of the blocks
  output->GetMetaData(blockIndex)->Set(vtkCompositeDataSet::NAME(), "Volumes");
  output->GetMetaData(blockIndex+1)->Set(vtkCompositeDataSet::NAME(), "Boundary");
  output->GetMetaData(blockIndex+2)->Set(vtkCompositeDataSet::NAME(), "Surfaces");
  output->GetMetaData(blockIndex+3)->Set(vtkCompositeDataSet::NAME(), "Neumann Sets");
  output->GetMetaData(blockIndex+4)->Set(vtkCompositeDataSet::NAME(), "Dirichlet Sets");

  smoab::GeomTag geom3Tag(3);
  smoab::GeomTag geom2Tag(2);
  smoab::GeomTag geom1Tag(2);
  smoab::NeumannTag neTag;
  smoab::DirichletTag diTag;

  smoab::Interface interface(this->FileName);
  this->CreateSubBlocks(volumeRoot, &interface, &geom3Tag);
  this->CreateSubBlocks(boundaryRoot, &interface, &geom3Tag, &geom2Tag);
  this->CreateSubBlocks(boundaryRoot, &interface, &geom3Tag, &geom1Tag);

  this->CreateSubBlocks(surfaceRoot, &interface, &geom2Tag);
  this->CreateSubBlocks(neumannRoot, &interface, &neTag);
  this->CreateSubBlocks(dirichletRoot, &interface, &diTag);


  vtkNew<vtkMultiBlockDataSet> shellRoot;
  output->SetBlock(blockIndex+5,shellRoot.GetPointer());
  output->GetMetaData(blockIndex+5)->Set(vtkCompositeDataSet::NAME(), "Volume Shell");
  this->ExtractShell(shellRoot,&interface,&geom3Tag);


  return 1;
}


//------------------------------------------------------------------------------
void vtkMoabReader::CreateSubBlocks(vtkNew<vtkMultiBlockDataSet> & root,
                                    smoab::Interface* interface,
                                    smoab::Tag const* parentTag,
                                    smoab::Tag const* extractTag)
{
  if(!extractTag)
    {
    extractTag = parentTag;
    }
  //basic premise: query the database for all tagged elements and create a new
  //multiblock elemenent for each
  smoab::DataSetConverter converter(*interface,extractTag);

  smoab::EntityHandle rootHandle = interface->getRoot();
  smoab::Range parents = interface->findEntityRootParents(rootHandle);
  smoab::Range dimEnts = interface->findEntitiesWithTag(*parentTag,
                                                       rootHandle);

  smoab::Range geomParents = smoab::intersect(parents,dimEnts);

  parents.clear(); //remove this range as it is unneeded
  dimEnts.clear();

  //now each item in range can be extracted into a different grid
  typedef smoab::Range::iterator iterator;
  vtkIdType index = 0;
  for(iterator i=geomParents.begin(); i != geomParents.end(); ++i)
    {
    vtkNew<vtkUnstructuredGrid> block;
    //fill the dataset with geometry and properties
    converter.fill(*i, block.GetPointer(),index);

    //only add it if we have cells found
    if(block->GetNumberOfCells() > 0)
      {
      root->SetBlock(index,block.GetPointer());
      std::string name = interface->name(*i);
      if(name.size() > 0)
        {
        root->GetMetaData(index)->Set(vtkCompositeDataSet::NAME(), name.c_str());
        }
      ++index;
      }
    }
}

//------------------------------------------------------------------------------
void vtkMoabReader::ExtractShell(vtkNew<vtkMultiBlockDataSet> & root,
                                smoab::Interface* interface,
                                smoab::Tag const* parentTag)
{

  smoab::EntityHandle rootHandle = interface->getRoot();
  smoab::Range parents = interface->findEntityRootParents(rootHandle);
  smoab::Range dimEnts = interface->findEntitiesWithTag(*parentTag,
                                                       rootHandle);

  smoab::Range geomParents = smoab::intersect(parents,dimEnts);

  smoab::ExtractShell shell(*interface,geomParents);

  vtkNew<vtkPolyData> output;
  shell.fill(output.GetPointer());

  root->SetBlock(0,output.GetPointer());
}

//------------------------------------------------------------------------------
void vtkMoabReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
