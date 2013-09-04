
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
#include <dax/opengl/testing/WindowBase.h>
#include <dax/opengl/TransferToOpenGL.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

namespace
{
typedef dax::Tuple<dax::Vector3,24> QuadCoordType;
typedef dax::Tuple<dax::Vector3,6> QuadNormalType;
typedef dax::Tuple<unsigned char,24> QuadColorType;

//The functor used to determine if a single cell passes the threshold reqs
template<typename T>
struct threshold_voxel : public dax::exec::internal::WorkletBase
{
  //we inherit from WorkletBase so that we can throw errors in the exec
  //env and the control env can find out why the worklet failed

  typedef dax::CellTagVoxel CellTag;
  dax::exec::internal::TopologyUniform Topology; //holds the cell connectivity

  //hold a portal so that we can get values in the exec env
  typedef typename dax::cont::ArrayHandle< T >::PortalConstExecution PortalType;
  PortalType Values;

  //holds the array of of what cells pass or fail the threshold reqs
  typedef dax::cont::ArrayHandle< int >::PortalExecution OutPortalType;
  OutPortalType PassesThreshold;

  T MinValue;
  T MaxValue;

  DAX_CONT_EXPORT
  threshold_voxel(const dax::cont::UniformGrid< >& grid,
                 const dax::cont::ArrayHandle<T>& values,
                 T min, T max,
                 dax::cont::ArrayHandle<int> passes):
  Topology(grid.PrepareForInput()), //upload grid topology to exec env
  Values(values.PrepareForInput()), //upload values to exec env
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
      const T value = this->Values.Get( verts[i] );
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
  PortalType ValidCellIds;
  PortalType CellCountPortal;


  //holds if this cell should be renderd
  typedef dax::cont::ArrayHandle< bool >::PortalExecution OutPortalType;
  OutPortalType ShouldRender;

  dax::Id3 Dims;

  DAX_CONT_EXPORT
  number_of_valid_neighhbors(const dax::cont::UniformGrid< >& grid,
                 const dax::cont::ArrayHandle<int>& valid_cell_ids,
                 const dax::cont::ArrayHandle<int>& cell_counts,
                 dax::cont::ArrayHandle<bool> should_render):
  ValidCellIds(valid_cell_ids.PrepareForInput()),
  CellCountPortal(cell_counts.PrepareForInput()),
  ShouldRender(should_render.PrepareForOutput( valid_cell_ids.GetNumberOfValues() )),
  Dims(dax::extentCellDimensions(grid.GetExtent()))
    {
    }

  DAX_EXEC_EXPORT
  void operator()( int index ) const
    {
    //find the mapping from new cell id to original id
    int cell_index = this->ValidCellIds.Get(index);

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

    int neighbor_count = !(j == 0)            && (CellCountPortal.Get(neighbors[0]) > 0);
    neighbor_count += !(i == this->Dims[0]-1) && (CellCountPortal.Get(neighbors[1]) > 0);
    neighbor_count += !(j == this->Dims[1]-1) && (CellCountPortal.Get(neighbors[2]) > 0);
    neighbor_count += !(i == 0)               && (CellCountPortal.Get(neighbors[3]) > 0);
    neighbor_count += !(k == 0)               && (CellCountPortal.Get(neighbors[4]) > 0);
    neighbor_count += !(k == this->Dims[2]-1) && (CellCountPortal.Get(neighbors[5]) > 0);

    //set us to 1 if we are going to generate faces, otherwise set to 0
    this->ShouldRender.Set(index,(neighbor_count != 6));
    }
};

template<typename T>
struct make_faces : public dax::exec::internal::WorkletBase
{
  typedef dax::CellTagVoxel CellTag;

  //hold a portal so that we can get values in the exec env
  typedef typename dax::cont::ArrayHandle< T >::PortalConstExecution ValuesPortalType;
  typedef typename dax::cont::ArrayHandle< int >::PortalConstExecution IntPortalType;

  typedef dax::cont::ArrayHandle< QuadCoordType >::PortalExecution OutCoordPortalType;
  typedef dax::cont::ArrayHandle< QuadNormalType >::PortalExecution OutNormPortalType;
  typedef dax::cont::ArrayHandle< QuadColorType >::PortalExecution OutColorPortalType;

  typedef dax::exec::internal::TopologyUniform InTopologyType;
  typedef typename dax::cont::UniformGrid< >::PointCoordinatesType::PortalConstExecution Vector3PortalType;

  ValuesPortalType Values;
  IntPortalType ValidCellIds;

  OutCoordPortalType Coords;
  OutNormPortalType Norms;
  OutColorPortalType Colors;

  InTopologyType InputTopology; //holds the input cell connectivity
  Vector3PortalType InputCoords; //coordinates of the input voxel

  DAX_CONT_EXPORT make_faces(const dax::cont::UniformGrid< >& grid,
                             const dax::cont::ArrayHandle<T>& values,
                             const dax::cont::ArrayHandle<int>& valid_cell_ids,
                             dax::cont::ArrayHandle<QuadCoordType> coords,
                             dax::cont::ArrayHandle<QuadNormalType> norms,
                             dax::cont::ArrayHandle<QuadColorType> colors):
  Values(values.PrepareForInput()),
  ValidCellIds(valid_cell_ids.PrepareForInput()),
  Coords(coords.PrepareForOutput(valid_cell_ids.GetNumberOfValues())),
  Norms(norms.PrepareForOutput(valid_cell_ids.GetNumberOfValues())),
  Colors(colors.PrepareForOutput(valid_cell_ids.GetNumberOfValues())),
  InputTopology(grid.PrepareForInput()),
  InputCoords(grid.GetPointCoordinates().PrepareForInput())
  {
  }

  DAX_EXEC_EXPORT
  void operator()( int index ) const
  {
    //we need to generate the following per voxel
    //1. 24 coordinates that describe the points of each quad
    //2. 6 normals one for each quad
    //3. 6 colors one for each quad

    const int vertices_for_faces[] =
      {
       0, 1, 5, 4, // face 0
       1, 2, 6, 5, // face 1
       2, 3, 7, 6, // face 2
       0, 4, 7, 3, // face 3
       0, 3, 2, 1, // face 4
       4, 5, 6, 7  // face 5
      };

    //find the mapping from new cell id to original id
    int cell_index = this->ValidCellIds.Get(index);

    dax::exec::CellVertices<CellTag> verts =
                            this->InputTopology.GetCellConnections(cell_index);

    //we need to compute the coordinates, normals and colors now
    //i think we might also need the min and max threshold values
  }

};
}

//we could template this class on the vector type, but why bother for this example?
class threshold_renderer :
                  public dax::opengl::testing::WindowBase<threshold_renderer>
{
private:
  //find the default device adapter
  typedef DAX_DEFAULT_DEVICE_ADAPTER_TAG AdapterTag;

  //Make it easy to call the DeviceAdapter with the right tag
  typedef dax::cont::internal::DeviceAdapterAlgorithm<AdapterTag> DeviceAdapter;

  dax::cont::UniformGrid<> InputGrid;
  dax::cont::ArrayHandle<float> InputScalars;
  float MinValue;
  float MaxValue;
  bool Dirty; //we need to rerender when true


  //gl array ids that hold the rendering info
  GLuint CoordHandle, ColorsHandle, NormalsHandle;

public:
  DAX_CONT_EXPORT threshold_renderer(const dax::cont::UniformGrid<>& grid,
                                     std::vector<float> values,
                                     float minv, float maxv)
  {
  //lets get the threshold ready to rock and roll
  this->InputGrid = grid;
  //this doesn't copy the data it just references the vectors internal data
  this->InputScalars = dax::cont::make_ArrayHandle(values);
  this->MinValue = minv;
  this->MaxValue = maxv;
  this->Dirty = true;
  }

  void contstruct_render_data()
  {
  this->Dirty = false;

  //schedule the thresholding on a per cell basis
  dax::cont::ArrayHandle<int> passesThreshold;
  threshold_voxel<float> tc(this->InputGrid,
                            this->InputScalars,
                            this->MinValue,
                            this->MaxValue,
                            passesThreshold);

  //since we are a uniform grid we can leverage the block scheduler in dax
  dax::Id3 dims = dax::extentCellDimensions(this->InputGrid.GetExtent());
  DeviceAdapter::Schedule( tc, dims );

  //extract only cells ids which pass the threshold
  dax::cont::ArrayHandle<int> validCellIndices;
  DeviceAdapter::StreamCompact( passesThreshold,
                                validCellIndices);

  //now that we have the good cell ids only, lets
  //see which of those cells we want to generate faces for.
  dax::cont::ArrayHandle<bool> shouldRenderCell;
  number_of_valid_neighhbors nv(this->InputGrid,
                                validCellIndices,
                                passesThreshold,
                                shouldRenderCell);
  DeviceAdapter::Schedule( nv,  validCellIndices.GetNumberOfValues() );

  //compact again on cells that we need to render
  dax::cont::ArrayHandle<int> cellsToRender;
  DeviceAdapter::StreamCompact( validCellIndices,
                                shouldRenderCell,
                                cellsToRender);
  validCellIndices.ReleaseResources();
  shouldRenderCell.ReleaseResources();

  //we can now generate the faces for the cells
  dax::cont::ArrayHandle<QuadCoordType> coords;
  dax::cont::ArrayHandle<QuadNormalType> normals;
  dax::cont::ArrayHandle<QuadColorType> colors;

  make_faces<float> mf(this->InputGrid,
                       this->InputScalars,
                       cellsToRender,
                       coords,
                       normals,
                       colors);

  DeviceAdapter::Schedule(mf, cellsToRender.GetNumberOfValues() );

  //now push that to opengl
  dax::opengl::TransferToOpenGL(coords, this->CoordHandle);
  dax::opengl::TransferToOpenGL(normals, this->NormalsHandle);
  dax::opengl::TransferToOpenGL(colors, this->ColorsHandle);
  }

  //called after opengl is inited
  DAX_CONT_EXPORT void PostInit()
  { this->contstruct_render_data(); }

  DAX_CONT_EXPORT void Display() { }

  DAX_CONT_EXPORT void Idle(){ }

  DAX_CONT_EXPORT void ChangeSize(int daxNotUsed(w), int daxNotUsed(h)) {}

  DAX_CONT_EXPORT void Key(unsigned char daxNotUsed(key), int daxNotUsed(x),
                           int daxNotUsed(y) ){ }

  DAX_CONT_EXPORT void SpecialKey(int daxNotUsed(key), int daxNotUsed(x),
                                  int daxNotUsed(y) ) { }

  DAX_CONT_EXPORT void Mouse(int daxNotUsed(button), int daxNotUsed(state),
                             int daxNotUsed(x), int daxNotUsed(y) ) {}

  DAX_CONT_EXPORT void MouseMove(int daxNotUsed(x), int daxNotUsed(y) ) {}
};

int main(int, char**)
{
  dax::cont::UniformGrid<> grid;
  grid.SetSpacing( dax::Vector3(0.004,0.004,0.004) );
  grid.SetExtent( dax::Id3(0,0,0), dax::Id3(255,255,255) );


  std::vector<float> data_store(26*256*256);

  //fill the vector with random numbers
  std::srand(42); //I like this seed :-)
  std::generate(data_store.begin(),data_store.end(),std::rand);

  const float sum = std::accumulate(data_store.begin(),data_store.end(),0.0f);
  const float average = sum / static_cast<float>(data_store.size());
  const float max = *(std::max_element(data_store.begin(),data_store.end()));

  //setup the input to threshold
  threshold_renderer renderer(grid,data_store,average,max);

  //init the render window
  renderer.Init("Threshold Example", 800, 600);

  //start drawing
  renderer.Start();
  return 0;
}
