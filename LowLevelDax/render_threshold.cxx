//Description:
//Threshold a voxel dataset where we only extract the exterior faces and
//pass those faces to openGL for rendering

#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayHandleCounting.h>
#include <dax/Extent.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/CellTag.h>
#include <dax/CellTraits.h>

//exec headers we need
#include <dax/exec/internal/WorkletBase.h> //required for error handling
#include <dax/exec/CellVertices.h>

//headers we need for opengl support
#include <dax/opengl/TransferToOpenGL.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

//The functor used to determine if a single cell passes the threshold reqs
template<class T>
struct threshold_voxel : public dax::exec::internal::WorkletBase
{
  //we inherit from WorkletBase so that we can throw errors in the exec
  //env and the control env can find out why the worklet failed

  typedef dax::CellTagVoxel CellTag;
  dax::exec::internal::TopologyUniform Topology; //holds the cell connectivity

  //hold a portal so that we can get values in the exec env
  typedef typename dax::cont::ArrayHandle< T >::PortalConstExecution PortalType;
  PortalType ValuePortal;

  //holds the array of of what cells pass or fail the threshold reqs
  typedef dax::cont::ArrayHandle< int >::PortalExecution OutPortalType;
  OutPortalType PassesThreshold;

  T MinValue;
  T MaxValue;

  DAX_CONT_EXPORT
  threshold_voxel(const dax::cont::UniformGrid< >& grid,
                 dax::cont::ArrayHandle<T> values, T min, T max,
                 dax::cont::ArrayHandle<int> passes):
  Topology(grid.PrepareForInput()), //upload grid topology to exec env
  ValuePortal(values.PrepareForInput()), //upload values to exec env
  MinValue(min),
  MaxValue(max),
  PassesThreshold(passes.PrepareForOutput( grid.GetNumberOfCells() ))
    {
    }

  DAX_EXEC_EXPORT
  void operator()( int cell_index ) const
    {
    //get all the point ids for the cell index
    dax::exec::CellVertices<CellTag> verts =
                                  this->Topology.GetCellConnections(cell_index);

    //for each vertice see if we are between the min and max which is
    //inclusive on both sides. We hint to the compiler that this is
    //a fixed size array by using NUM_VERTICES. This can be easily
    //unrolled if needed
    int valid = 1;
    for(int i=0; i < dax::CellTraits<CellTag>::NUM_VERTICES; ++i)
      {
      const T value = this->ValuePortal.Get( verts[i] );
      valid &= value >= this->MinValue && value <= this->MaxValue;
      }
    this->PassesThreshold.Set(cell_index,valid);
    }
};

//this struct will count the number of cell neighbors that are valid
//based on that we can determine the number of faces that need to be shown
struct number_of_valid_neighhbors: public dax::exec::internal::WorkletBase
{
  //store the cell type that we are working on
  typedef dax::CellTagVoxel CellTag;

  //hold how many cells each cell will output
  typedef dax::cont::ArrayHandle< int >::PortalConstExecution PortalType;
  PortalType CellCountPortal;

  //hold how many faces this cell will generate
  typedef dax::cont::ArrayHandle< int >::PortalExecution OutPortalType;
  OutPortalType FaceCountPortal;

  dax::Id3 Dims;

  DAX_CONT_EXPORT
  number_of_valid_neighhbors(const dax::cont::UniformGrid< >& grid,
                 dax::cont::ArrayHandle<int> cellCounts,
                 dax::cont::ArrayHandle<int> faces):
  CellCountPortal(cellCounts.PrepareForInput()),
  FaceCountPortal(faces.PrepareForOutput( grid.GetNumberOfCells() )),
  Dims(dax::extentCellDimensions(grid.GetExtent()))
    {
    }

  DAX_EXEC_EXPORT
  void operator()( int cell_index ) const
    {
    //compute the neighbors of this cell
    int neighbors[6] = { cell_index - this->Dims[0],
                         cell_index + 1,
                         cell_index + this->Dims[0],
                         cell_index - 1,
                         cell_index - this->Dims[0] * this->Dims[1],
                         cell_index + this->Dims[0] * this->Dims[1]};

    //determine if we are a cell that is on the boundary of any sides
    const int i = cell_index % this->Dims[0];
    const int j = (cell_index  / this->Dims[0]) % this->Dims[1];
    const int k = cell_index / (this->Dims[0] * this->Dims[1]);

    int neighbor_count = !(j == 0)            && CellCountPortal.Get(neighbors[0]);
    neighbor_count += !(i == this->Dims[0]-1) && CellCountPortal.Get(neighbors[1]);
    neighbor_count += !(j == this->Dims[1]-1) && CellCountPortal.Get(neighbors[2]);
    neighbor_count += !(i == 0)               && CellCountPortal.Get(neighbors[3]);
    neighbor_count += !(k == 0)               && CellCountPortal.Get(neighbors[4]);
    neighbor_count += !(k == this->Dims[2]-1) && CellCountPortal.Get(neighbors[5]);

    this->FaceCountPortal.Set(cell_index,neighbor_count);
    }
};


// return true if the cell will actually generate geometry.
struct is_exterior_cell
{
DAX_EXEC_EXPORT bool operator() (int count) const
  { return count != 6; }
};

struct make_faces
{

};



int main(int, char**)
{
  return 0;
}