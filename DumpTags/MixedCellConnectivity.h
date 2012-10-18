#ifndef __smoab_MixedCellConnectivity_h
#define __smoab_MixedCellConnectivity_h

#include "vtkCellType.h"
#include <algorithm>

namespace
{

template<int N> struct QuadratricOrdering{};

template<> struct QuadratricOrdering<VTK_QUADRATIC_WEDGE>
{
  static const int NUM_VERTS = 15;
  void reorder(vtkIdType* connectivity) const
  {
    std::swap_ranges(connectivity+9,connectivity+12,connectivity+12);
  }
};

template<> struct QuadratricOrdering<VTK_TRIQUADRATIC_HEXAHEDRON>
{
  static const int NUM_VERTS = 27;
  void reorder(vtkIdType* connectivity) const
  {
    std::swap_ranges(connectivity+12,connectivity+16,connectivity+16);

    //move 20 to 22
    //move 22 to 23
    //move 23 to 20

    //swap 20 with 22
    std::swap(connectivity[20],connectivity[23]);

    //swap 22 with 23
    std::swap(connectivity[22],connectivity[23]);
  }
};

template<typename QuadraticOrdering>
void FixQuadraticIdOrdering(vtkIdType* connectivity, vtkIdType numCells,
                            QuadraticOrdering& ordering)
{
  //skip the first index that holds the length of the cells
  //if we skip it once here, and than properly increment it makes the code
  //far easier
  connectivity+=1;
  for(vtkIdType i=0; i < numCells; ++i)
    {
    ordering.reorder(connectivity);
    connectivity += ordering.NUM_VERTS + 1;
    }
}
}

namespace smoab
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
    const std::size_t cellSize=cells.size();
    while(count != cellSize)
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

      //while all these cells are contiously of the same type,
      //quadric hexs in vtk have 20 points, but moab has 21 so we
      //need to store this difference
      int numVTKVerts = numVerts;
      int vtkCellType = smoab::vtkCellType(type,numVTKVerts);

      RunLengthInfo info = { vtkCellType, numVerts, (numVerts-numVTKVerts), iterationCount };
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
      const int numUnusedPoints = (*i).numUnusedVerts;
      if(numUnusedPoints==0)
        {
        const int connLength = (*i).numCells * (*i).numVerts;
        std::copy(*c,*c+connLength,std::back_inserter(output));
        }
      else
        {
        //we have cell connectivity that we need to skip,
        //so we have to manual copy each cells connectivity
        const int size = (*i).numCells;
        const int numPoints = (*i).numVerts;
        for(int j=0; j < size; ++j)
          {
          std::copy(*c,*c+numPoints,std::back_inserter(output));
          }
        c+=numPoints + (*i).numUnusedVerts;
        }

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

        //skip any extra unused points, which is currnetly only
        //the extra center point in moab quadratic hex
        moabConnectivity+=(*i).numUnusedVerts;

        currentVtkConnectivityIndex += numVerts+1;
        cellArray += numVerts+1;
        }

      //For Tri-Quadratic-Hex and Quadratric-Wedge Moab and VTK
      //Differ on the order of the edge ids. For wedge we need to swap
      //indices 9,10,11 with 12,13,14 for each cell. For Hex we sawp
      //12,13,14,15 with 16,17,18,19
      int vtkCellType = (*i).type;
      vtkIdType* connectivity = cellArray - (numCells * (numVerts+1));
      if(vtkCellType == VTK_TRIQUADRATIC_HEXAHEDRON)
        {
        ::QuadratricOrdering<VTK_TRIQUADRATIC_HEXAHEDRON> newOrdering;
        ::FixQuadraticIdOrdering(connectivity, numCells, newOrdering);
        }
      else if(vtkCellType == VTK_QUADRATIC_WEDGE)
        {
        ::QuadratricOrdering<VTK_QUADRATIC_WEDGE> newOrdering;
        ::FixQuadraticIdOrdering(connectivity, numCells, newOrdering);
        }

      cellLocations += numCells;
      cellTypes += numCells;
      }

    }

private:
  std::vector<EntityHandle*> Connectivity;
  std::vector<EntityHandle> UniquePoints;

  struct RunLengthInfo{ int type; int numVerts; int numUnusedVerts; int numCells; };
  std::vector<RunLengthInfo> Info;

  typedef std::vector<EntityHandle>::const_iterator EntityConstIterator;
  typedef std::vector<EntityHandle*>::const_iterator ConnConstIterator;
  typedef std::vector<RunLengthInfo>::const_iterator InfoConstIterator;
};
}
#endif // __smoab_MixedCellConnectivity_h
