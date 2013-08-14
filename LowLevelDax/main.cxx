
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/CellTag.h>
#include <dax/CellTraits.h>

//exec headers we need
#include <dax/exec/internal/WorkletBase.h> //required for error handling
#include <dax/exec/CellVertices.h>

#include <iostream>

//The functor used to determine if a single cell passes the threshold reqs
template<class GridType, class T>
struct threshold_cell : public dax::exec::internal::WorkletBase
{
  //we inherit from WorkletBase so that we can throw errors in the exec
  //env and the control env can find out why the worklet failed

  //store the cell type that we are working on
  typedef typename GridType::CellTag CellTag;

  //determine the topology type that we need in the exec env
  typedef typename GridType::TopologyStructConstExecution TopologyType;
  TopologyType Topology; //holds the cell connectivity

  //hold a portal so that we can get values in the exec env
  typedef typename dax::cont::ArrayHandle< T >::PortalConstExecution PortalType;
  PortalType ValuePortal;

  //holds the array of of what cells pass or fail the threshold reqs
  typedef typename dax::cont::ArrayHandle< int >::PortalExecution OutPortalType;
  OutPortalType PassesThreshold;

  T MinValue;
  T MaxValue;

  DAX_CONT_EXPORT
  threshold_cell(const GridType& grid,
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


//this struct will do the cell sub-setting by being given the input topology
//and the cell ids that need to be added to the new output grid
template<class GridType, class OutGridType >
struct cell_subset: public dax::exec::internal::WorkletBase
{
  //we inherit from WorkletBase so that we can throw errors in the exec
  //env and the control env can find out why the worklet failed
  //store the cell type that we are working on
  typedef typename GridType::CellTag CellTag;

  //determine the topology type for the input grid
  typedef typename GridType::TopologyStructConstExecution InTopologyType;
  InTopologyType InputTopology; //holds the input cell connectivity

  //determine the topology type for the output grid
  typedef typename OutGridType::TopologyStructExecution OutTopologyType;
  OutTopologyType OutputTopology; //holds the output cell connectivity

  typedef typename dax::cont::ArrayHandle< int >::PortalConstExecution PortalType;
  PortalType PermutationPortal;

  DAX_CONT_EXPORT
  cell_subset(const GridType& grid,
              OutGridType& outGrid,
              dax::cont::ArrayHandle<int> permutationIndices):
  InputTopology(grid.PrepareForInput()),
  OutputTopology(outGrid.PrepareForOutput(
                            permutationIndices.GetNumberOfValues())),
  PermutationPortal(permutationIndices.PrepareForInput())
    {
    }

  DAX_EXEC_EXPORT
  void operator()( int cell_index ) const
    {
    dax::exec::CellVertices<CellTag> verts =
                            this->InputTopology.GetCellConnections(cell_index);

    const int offset = dax::CellTraits<CellTag>::NUM_VERTICES * cell_index;
    for(int i=0; i < dax::CellTraits<CellTag>::NUM_VERTICES; ++i)
      {
      this->OutputTopology.CellConnections.Set(offset+i,verts[i]);
      }
    }
};


template<class GridType ,class T>
void ThresholdExample(GridType grid, std::vector<T> &array,
                      T minValue, T  maxValue)
{
  const int numCells = grid.GetNumberOfCells();

  //find the default device adapter
  typedef DAX_DEFAULT_DEVICE_ADAPTER_TAG AdapterTag;

  //Make it easy to call the DeviceAdapter with the right tag
  typedef dax::cont::internal::DeviceAdapterAlgorithm<AdapterTag> DeviceAdapter;


  //container and portal used for counting array handles
  typedef dax::cont::internal::ArrayContainerControlTagCounting CountingTag;
  typedef dax::cont::internal::ArrayPortalCounting<int> CountingPortalType;


  //make a handle to the std::vector, this actually doesn't copy any memory
  //but waits for something to call PrepareForInput or PrepareForOutput before
  //moving the memory to cuda/tbb if required
  dax::cont::ArrayHandle<T> arrayHandle = dax::cont::make_ArrayHandle(array);


  //schedule the thresholding on a per cell basis
  dax::cont::ArrayHandle<int> passesThreshold;
  threshold_cell<GridType,T> tc(grid, arrayHandle, minValue, maxValue,
                                passesThreshold);

  DeviceAdapter::Schedule( tc,  numCells );

  dax::cont::ArrayHandle<int> onlyGoodCellIds;
  const dax::Id numNewCells =
              DeviceAdapter::ScanInclusive( passesThreshold, onlyGoodCellIds );

  dax::cont::ArrayHandle<int> cellUpperBounds;
  cellUpperBounds.PrepareForOutput(numNewCells); //allocate

  //create the counting array handle for the upper bounds
  CountingPortalType cportal(0,numNewCells);
  dax::cont::ArrayHandle<int,CountingTag> countingHandle(cportal);

  DeviceAdapter::UpperBounds( onlyGoodCellIds,
                              countingHandle,
                              cellUpperBounds );

  //now that we have the good cell ids only
  //lets extract the topology for those cells by calling cell_subset

  //first step is to find the cell type for the output grid. Since
  //the input grid can be any cell type, we need to find the
  //CanonicalCellTag for the cell type to determine what kind of cell
  //it will be in an unstructured grid
  typedef dax::CellTraits<typename GridType::CellTag> CellTraits;
  typedef typename CellTraits::CanonicalCellTag OutCellType;
  typedef dax::cont::ArrayContainerControlTagBasic CellContainerTag;


  //the second step is to copy the grid point coordinates in full
  //as we are doing cell sub-setting really. The first step is to
  //get the container tag type from the input point coordinates
  typedef typename GridType::PointCoordinatesType PointCoordsArrayHandle;
  typedef typename PointCoordsArrayHandle::ArrayContainerControlTag
                                                       PointContainerTag;

  //now determine the out unstructured grid type
  typedef dax::cont::UnstructuredGrid<OutCellType,
                CellContainerTag, PointContainerTag> OutGridType;

  OutGridType outGrid;
  outGrid.SetPointCoordinates(grid.GetPointCoordinates());

  // //now time to do the actual cell sub-setting
  cell_subset<GridType,OutGridType> cs(grid,outGrid,cellUpperBounds);
  DeviceAdapter::Schedule(cs,cellUpperBounds.GetNumberOfValues());


  std::cout << "Input Grid number of cells: " << numCells << std::endl;
  std::cout << "Output Grid number of cells: " << outGrid.GetNumberOfCells() << std::endl;

};

int main()
{

  //load up a uniform grid and point based array and threshold

  dax::cont::UniformGrid<> grid;
  grid.SetExtent( dax::Id3(0,0,0), dax::Id3(4,4,4) );

  std::vector<float> data_store(5*5*5,25);
  float min=0, max=100;

  ThresholdExample(grid,data_store,min,max);

}