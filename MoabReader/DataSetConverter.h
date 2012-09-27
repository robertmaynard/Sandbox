#ifndef __smoab_DataSetConverter_h
#define __smoab_DataSetConverter_h

#include "SimpleMoab.h"
#include "CellTypeToType.h"

#include <vtkPoints.h>
#include <vtkNew.h>
#include <vtkUnstructuredGrid.h>

#include <algorithm>
namespace smoab
{

namespace detail
{

class MixedCellConnectivity
{
public:
  MixedCellConnectivity():
    Connectivity(),
    UniqueIds()
  {
  }

  //----------------------------------------------------------------------------
  void add(int vtkCellType, int numVerts, smoab::EntityHandle* conn, int numCells)
    {
    RunLengthInfo info = { vtkCellType, numVerts, numCells };
    this->Info.push_back(info);
    this->Connectivity.push_back(conn);
    }

  //----------------------------------------------------------------------------
  void compactIdsAndSet(vtkUnstructuredGrid* grid)
    {
    //converts all the ids to be ordered starting at zero, and also
    //keeping the orginal logical ordering. Stores the result of this
    //operation in the unstrucutred grid that is passed in

    //lets determine the total length of the connectivity
    vtkIdType connectivityLength = 0;
    vtkIdType totalNumCells = 0;
    for(InfoConstIterator i = this->Info.begin();
        i != this->Info.end();
        ++i)
      {
      connectivityLength += (*i).numCells * (*i).numVerts;
      totalNumCells += (*i).numCells;
      }

    this->UniqueIds.reserve(connectivityLength);
    this->copyConnectivity( );



    std::sort(this->UniqueIds.begin(),this->UniqueIds.end());

    EntityIterator newEnd = std::unique(this->UniqueIds.begin(),
                                        this->UniqueIds.end());

    const std::size_t newSize = std::distance(this->UniqueIds.begin(),newEnd);
    this->UniqueIds.resize(newSize); //release unneeded space

    this->fillGrid(grid,totalNumCells, connectivityLength);
    }

  //----------------------------------------------------------------------------
  smoab::Range uniquePointIds()
    {
    //from the documentation a reverse iterator is the fastest way
    //to insert into a range. that is why mixConn.begin is really rbegin, and
    //the same with end
    moab::Range result;
    std::copy(UniqueIds.rbegin(),UniqueIds.rend(),moab::range_inserter(result));
    return result;
    }
private:

  void copyConnectivity()
    {

    }

  //----------------------------------------------------------------------------
  void fillGrid(vtkUnstructuredGrid* grid,
                int numCells,
                int numConnectivity) const
    {
    //correct the connectivity size to account for the vtk padding
    int vtkConnectivity = numCells + numConnectivity;


    //for each item in the connectivy add it to the grid!
    }

  std::vector<EntityHandle*> Connectivity;
  std::vector<EntityHandle> UniqueIds;

  struct RunLengthInfo{ int type; int numVerts; int numCells; };
  std::vector<RunLengthInfo> Info;

  typedef std::vector<EntityHandle>::iterator EntityIterator;
  typedef std::vector<RunLengthInfo>::const_iterator InfoConstIterator;
};
}

class DataSetConverter
{
  smoab::Interface Interface;
  moab::Interface* Moab;

public:
  DataSetConverter(const smoab::Interface& interface ):
    Interface(interface),
    Moab(interface.Moab)
    {
    }

  //----------------------------------------------------------------------------
  bool fill(const smoab::EntityHandle& entity, vtkUnstructuredGrid* grid) const
    {
    smoab::Range pointRange = this->addCells(entity,grid);

    vtkNew<vtkPoints> newPoints;
    this->addCoordinates(pointRange,newPoints.GetPointer());
    grid->SetPoints(newPoints.GetPointer());

    return true;
    }

  //----------------------------------------------------------------------------
  //given a range of entity types, add them to an unstructured grid
  //we return a Range object that holds all the point ids that we used
  //which is sorted and only has unique values.
  ///we use entity types so that we can determine vtk cell type :(
  moab::Range addCells(moab::EntityHandle root,
                       vtkUnstructuredGrid* grid) const
    {

    moab::Range cells = this->Interface.findEntitiesWithDimension(root,
                          this->Moab->dimension_from_handle(root));


    detail::MixedCellConnectivity mixConn;

    int count = 0;
    while(count != cells.size())
      {
      EntityHandle* connectivity;
      int numVerts=0, iterationCount=0;
      //use the highly efficent calls, since we know that are of the same dimension
      this->Moab->connect_iterate(cells.begin()+count,
                                  cells.end(),
                                  connectivity,
                                  numVerts,
                                  iterationCount);
      count += iterationCount;
      //if we didn't read anything, break!
      if(iterationCount == 0)
        {
        break;
        }

      //identify the cell type that we currently have,
      //store that along with the connectivity in a temp storage vector
      moab::EntityType type = this->Moab->type_from_handle(connectivity[0]);
      int vtkCellType = smoab::vtkCellType(type,numVerts); //have vtk cell type, for all these cells

      mixConn.add(vtkCellType,numVerts,connectivity,iterationCount);
      }

    //now that mixConn has all the cells properly stored, lets fixup
    //the ids so that they start at zero and keep the same logical ordering
    //as before.
    mixConn.compactIdsAndSet(grid);


    return mixConn.uniquePointIds();
    }

  //----------------------------------------------------------------------------
  void addCoordinates(smoab::Range pointEntities, vtkPoints* pointContainer) const
    {
    //since the smoab::range are always unique and sorted
    //we can use the more efficient coords_iterate
    //call in moab, which returns moab internal allocated memory
    pointContainer->SetDataTypeToDouble();
    pointContainer->SetNumberOfPoints(pointEntities.size());

    //need a pointer to the allocated vtkPoints memory so that we
    //don't need to use an extra copy and we can bypass all vtk's check
    //on out of bounds
    double *rawPoints = static_cast<double*>(pointContainer->GetVoidPointer(0));

    double *x,*y,*z;
    int count=0;
    while(count != pointEntities.size())
      {
      int iterationCount=0;
      this->Moab->coords_iterate(pointEntities.begin()+count,
                               pointEntities.end(),
                               x,y,z,
                               iterationCount);
      count+=iterationCount;

      //copy the elements we found over to the vtkPoints
      for(int i=0; i < iterationCount; ++i, rawPoints+=3)
        {
        rawPoints[i] = x[i];
        rawPoints[i+1] = y[i];
        rawPoints[i+2] = z[i];
        }
      }
    }

};

}
#endif // __smoab_DataSetConverter_h
