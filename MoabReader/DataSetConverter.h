#ifndef __smoab_DataSetConverter_h
#define __smoab_DataSetConverter_h

#include "SimpleMoab.h"
#include "CellTypeToType.h"

#include <vtkCellArray.h>
#include <vtkIdTypeArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>

#include <algorithm>
namespace smoab
{

namespace detail
{

class MixedCellConnectivity
{
public:
  MixedCellConnectivity(smoab::Range const& cells, moab::Interface* moab):
    Connectivity(),
    UniquePoints(),
    Info()
    {
    int count = 0;
    while(count != cells.size())
      {
      EntityHandle* connectivity;
      int numVerts=0, iterationCount=0;
      //use the highly efficent calls, since we know that are of the same dimension
      moab->connect_iterate(cells.begin()+count,
                            cells.end(),
                            connectivity,
                            numVerts,
                            iterationCount);
      //if we didn't read anything, break!
      if(iterationCount == 0)
        {
        break;
        }

      //identify the cell type that we currently have,
      //store that along with the connectivity in a temp storage vector
      const moab::EntityType type = moab->type_from_handle(*cells.begin()+count);
      //all these cells are contiously of the same type
      int vtkCellType = smoab::vtkCellType(type,numVerts);

      RunLengthInfo info = { vtkCellType, numVerts, iterationCount };
      this->Info.push_back(info);
      this->Connectivity.push_back(connectivity);

      count += iterationCount;
      }
    }

  //----------------------------------------------------------------------------
  void compactIds(vtkIdType& numCells, vtkIdType& connectivityLength)
    {
    //converts all the ids to be ordered starting at zero, and also
    //keeping the orginal logical ordering. Stores the result of this
    //operation in the unstrucutred grid that is passed in

    //lets determine the total length of the connectivity
    connectivityLength = 0;
    numCells = 0;
    for(InfoConstIterator i = this->Info.begin();
        i != this->Info.end();
        ++i)
      {
      connectivityLength += (*i).numCells * (*i).numVerts;
      numCells += (*i).numCells;
      }

    this->UniquePoints.reserve(connectivityLength);

    this->copyConnectivity(this->UniquePoints);
    std::sort(this->UniquePoints.begin(),this->UniquePoints.end());

    typedef std::vector<EntityHandle>::iterator EntityIterator;
    EntityIterator newEnd = std::unique(this->UniquePoints.begin(),
                                        this->UniquePoints.end());

    const std::size_t newSize = std::distance(this->UniquePoints.begin(),newEnd);
    this->UniquePoints.resize(newSize);
    }

  //----------------------------------------------------------------------------
  void moabPoints(smoab::Range& range) const
    {
    //from the documentation a reverse iterator is the fastest way
    //to insert into a range.
    std::copy(this->UniquePoints.rbegin(),
              this->UniquePoints.rend(),
              moab::range_inserter(range));
    }

  //----------------------------------------------------------------------------
  //copy the connectivity from the moab held arrays to the user input vector
  void copyConnectivity(std::vector<EntityHandle>& output) const
    {
    //walk the info to find the length of each sub connectivity array,
    //and insert them into the vector, ordering is implied by the order
    //the connecitivy sub array are added to this class
    ConnConstIterator c = this->Connectivity.begin();
    for(InfoConstIterator i = this->Info.begin();
        i != this->Info.end();
        ++i,++c)
      {
      //remember our Connectivity is a vector of pointers whose
      //length is held in the info vector.
      const int connLength = (*i).numCells * (*i).numVerts;
      std::copy(*c,*c+connLength,std::back_inserter(output));
      }
    }

  //copy the information from this contianer to a vtk cell array, and
  //related lookup information
  void copyToVtkCellInfo(vtkIdType* cellArray,
                         vtkIdType* cellLocations,
                         unsigned char* cellTypes) const
    {
    vtkIdType currentVtkConnectivityIndex = 0;
    ConnConstIterator c = this->Connectivity.begin();
    for(InfoConstIterator i = this->Info.begin();
        i != this->Info.end();
        ++i, ++c)
      {
      //for this group of the same cell type we need to fill the cellTypes
      const int numCells = (*i).numCells;
      const int numVerts = (*i).numVerts;

      std::fill_n(cellTypes,
                  numCells,
                  static_cast<unsigned char>((*i).type));

      //for each cell in this collection that have the same type
      //grab the raw array now, so we can properly increment for each vert in each cell
      EntityHandle* moabConnectivity = *c;
      for(int j=0;j < numCells; ++j)
        {
        cellLocations[j]= currentVtkConnectivityIndex;

        //cell arrays start and end are different, since we
        //have to account for element that states the length of each cell
        cellArray[0]=numVerts;

        for(int k=0; k < numVerts; ++k, ++moabConnectivity )
          {
          //this is going to be a root of some failures when we start
          //reading really large datasets under 32bit.


          //fyi, don't use a range ds for unique points, distance
          //function is horribly slow they need to override it
          EntityConstIterator result = std::lower_bound(
                                         this->UniquePoints.begin(),
                                         this->UniquePoints.end(),
                                         *moabConnectivity);
          std::size_t newId = std::distance(this->UniquePoints.begin(),
                                            result);
          cellArray[k+1] = static_cast<vtkIdType>(newId);
          }

        currentVtkConnectivityIndex += numVerts+1;
        cellArray += numVerts+1;
        }

      cellLocations += numCells;
      cellTypes += numCells;
      }

    }

private:

  std::vector<EntityHandle*> Connectivity;
  std::vector<EntityHandle> UniquePoints;

  struct RunLengthInfo{ int type; int numVerts; int numCells; };
  std::vector<RunLengthInfo> Info;

  typedef std::vector<EntityHandle>::const_iterator EntityConstIterator;
  typedef std::vector<EntityHandle*>::const_iterator ConnConstIterator;
  typedef std::vector<RunLengthInfo>::const_iterator InfoConstIterator;
};
}

class DataSetConverter
{
  const smoab::Interface& Interface;
  moab::Interface* Moab;
  smoab::GeomTag GeomDimTag;

public:
  DataSetConverter(const smoab::Interface& interface,const smoab::GeomTag& dim):
    Interface(interface),
    Moab(interface.Moab),
    GeomDimTag(dim)
    {
    }

  //----------------------------------------------------------------------------
  bool fill(const smoab::EntityHandle& entity,
            vtkUnstructuredGrid* grid) const
    {
    //create a helper datastructure which can determines all the unique point ids
    //and converts moab connecitvity info to vtk connectivity
    moab::Range cells = this->Interface.findEntitiesWithDimension(entity,
                                                                  this->GeomDimTag.value());
    detail::MixedCellConnectivity mixConn(cells,this->Moab);

    //now that mixConn has all the cells properly stored, lets fixup
    //the ids so that they start at zero and keep the same logical ordering
    //as before.
    vtkIdType numCells, connLen;
    mixConn.compactIds(numCells,connLen);
    this->fillGrid(mixConn,grid,numCells,connLen);


    this->readCellProperties(cells,grid);

    smoab::Range moabPoints;
    mixConn.moabPoints(moabPoints);

    vtkNew<vtkPoints> newPoints;
    this->addCoordinates(moabPoints,newPoints.GetPointer());
    grid->SetPoints(newPoints.GetPointer());

    this->readPointProperties(moabPoints,grid);

    return true;
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
    this->Moab->get_coords(pointEntities,rawPoints);
    }

private:

  //----------------------------------------------------------------------------
  void readPointProperties(smoab::Range const& pointEntities,
                           vtkUnstructuredGrid* grid) const
    {
    //we want all the entities for the points so we find a tag, we know
    //it is a point tag

    }

  //----------------------------------------------------------------------------
  void readCellProperties(smoab::Range const& mixedCells,
                          vtkUnstructuredGrid* grid) const
    {
    //mixed cells can give use all the entities in a vector that are cell ids

    }

  //----------------------------------------------------------------------------
  void fillGrid(detail::MixedCellConnectivity const& mixedCells,
                vtkUnstructuredGrid* grid,
                vtkIdType numCells,
                vtkIdType numConnectivity) const
    {
    //correct the connectivity size to account for the vtk padding
    const vtkIdType vtkConnectivity = numCells + numConnectivity;

    vtkNew<vtkIdTypeArray> cellArray;
    vtkNew<vtkIdTypeArray> cellLocations;
    vtkNew<vtkUnsignedCharArray> cellTypes;

    cellArray->SetNumberOfValues(vtkConnectivity);
    cellLocations->SetNumberOfValues(numCells);
    cellTypes->SetNumberOfValues(numCells);

    vtkIdType* rawArray = static_cast<vtkIdType*>(cellArray->GetVoidPointer(0));
    vtkIdType* rawLocations = static_cast<vtkIdType*>(cellLocations->GetVoidPointer(0));
    unsigned char* rawTypes = static_cast<unsigned char*>(cellTypes->GetVoidPointer(0));

    mixedCells.copyToVtkCellInfo(rawArray,rawLocations,rawTypes);

    vtkNew<vtkCellArray> cells;
    cells->SetCells(numCells,cellArray.GetPointer());
    grid->SetCells(cellTypes.GetPointer(),
                   cellLocations.GetPointer(),
                   cells.GetPointer(),
                   NULL,NULL);
    }
};

}
#endif // __smoab_DataSetConverter_h
