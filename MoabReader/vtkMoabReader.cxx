#include "vtkMoabReader.h"

#include "SimpleMoab.h"

#include "vtkNew.h"
#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkUnstructuredGrid.h"

namespace detail
{

  void createGrid(const smoab::Interface& interface,
            int dimensonality,
            smoab::EntityHandle entity,
            vtkNew<vtkUnstructuredGrid> &grid)
  {
    //we need to walk everything below the entity and get all the volume elements
    //from those volume elements
    smoab::EntityType start;
    smoab::EntityType end;
    switch(dimensonality)
      {
      case 2:
        start = moab::MBEDGE;
        end = moab::MBTET;
        break;
      case 3:
        start = moab::MBTET;
        end = moab::MBENTITYSET;
        break;
      default:
        //invalid dimension to load
        return;
        break;
      }


    moab::Range pointRange = interface.addCells(start,end,entity,grid.GetPointer());

    //right side return since we want to allocate right into the vtkPoints
    //since this is heavy data
    vtkNew<vtkPoints> points;
    interface.addCoordinates(pointRange,points.GetPointer());
  }
}

vtkStandardNewMacro(vtkMoabReader)
//------------------------------------------------------------------------------
vtkMoabReader::vtkMoabReader()
  {
  this->SetNumberOfInputPorts(0);
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

  smoab::Range parents = interface.findEntityRootParents(interface.getRoot());
  smoab::Range geom3dParents = parents.subset_by_dimension(dimensionality);
  parents.clear(); //remove this range as it is unneeded

  //now each item in range can be extracted into a different grid
  typedef smoab::Range::iterator iterator;
  vtkIdType index = 0;
  for(iterator i=geom3dParents.begin(); i != geom3dParents.end(); ++i, ++index)
    {
    vtkNew<vtkUnstructuredGrid> block;
    root->SetBlock(index,block.GetPointer());
    detail::createGrid(interface,dimensionality,*i,block);
    }
}

//------------------------------------------------------------------------------
void vtkMoabReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

