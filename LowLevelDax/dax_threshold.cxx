//Description:
//Basic Threshold of a dataset.

#include <dax/CellTag.h>
#include <dax/CellTraits.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

//headers needed for testing
#include <dax/cont/testing/TestingGridGenerator.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>


/*
################################################################################
####                   DAX Scheduler Infrastructure                         ####
################################################################################

  The Dax scheduler infrastructure and how to extend it is documented
  in the following link:
      http://www.daxtoolkit.org/index.php/Understanding_Schedule
      https://gist.github.com/robertmaynard/6038997 (markdown format)

  This documentation is how to add new Worklet Signature types like Field(),
  or Topology, and is considered to be an advanced topic


*/

/*
################################################################################
####                   DAX Scheduler Notes                                  ####
################################################################################

  We are currently in the process of revising the scheduler infastructure.
  Mainly dealing with how complicated it is to create a new scheduler class.

  The updated proposal of the new system can be found at:
    http://www.daxtoolkit.org/index.php/Simplified_Scheduler_Objects

*/

    /*
################################################################################
####                   Finding Dax Files                                    ####
################################################################################

  Dax tries to maintain a 1 to 1 correspondence between a classes namespace and
  where it will be located on the file system.

  So the class dax::cont::ArrayHandle will be located at dax/cont/ArrayHandle.h

*/

namespace worklet
{

template<typename ValueType>
class ThresholdClassify : public dax::exec::WorkletMapCell
{
public:
  /*
  Control Signature.
  Each argument in the control signature maps to a required argument
  that the user must pass to the scheduler. So for this example
  the user must specify 3 arguments to the control side. The first
  being geometry, the second being a point field, and the third being
  an output cell field

  These Control Signature classes are all located under:
    dax/cont/arg/ Topology.h, Field.h

  The qualifiers (In,Out) and the Domains (Cells,Points) are located under:
    dax/cont/sig/Tag.h
  */
  typedef void ControlSignature( Topology, Field(Point), Field(Out) );

  /*
  Execution Signature
  The execution signature maps the control signature values to what
  the operator() wants. So in this case we state that we don't need the
  Topology argument so drop it ( note no _1 ), we want the Field(Point) to be
  passed as the first argument, and we are going to return a value that
  will be placed into the Field(Out) ( see the _3 as the return value for
  the signature )
  */
  typedef _3 ExecutionSignature( _2 );

  DAX_CONT_EXPORT
  ThresholdClassify(ValueType thresholdMin, ValueType thresholdMax)
    : ThresholdMin(thresholdMin), ThresholdMax(thresholdMax) {  }


  /*
    Execution invocation of worklet

    We have templated the signature based on the topology type, which
    is only possible since we stated that the first argument would be of
    type topology. Without that information we can't

  */
  template<class CellTag>
  DAX_EXEC_EXPORT
  dax::Id operator()(
      const dax::exec::CellField<ValueType,CellTag> &values) const
  {
    /* simplified form that doesn't handle vector values
       if you need threshold to handle the use case where ValueType
       it self is a Tuple or fixed size array which doesn't implement
       the <= operator, you will need to implement a more robust algorithm.
       See dax/worklet/threshold for how this can be done
    */

    int valid = 1;
    for(unsigned int i=0; i < dax::CellTraits<CellTag>::NUM_VERTICES; ++i)
      { valid &= (values[i] >= ThresholdMin && values[i] <= ThresholdMax); }
    return valid;
  }
private:
  ValueType ThresholdMin;
  ValueType ThresholdMax;
};

class ThresholdTopology : public dax::exec::WorkletGenerateTopology
{
public:
  /*
  Control Signature.
  For this control signature we state we have to topology arguments required
  the first will be input topology and the second will be output topology.
  It should be noted that all worklets that inherit from WorkletGenerateTopology
  are required to have an input and ouput topology at minimum

  These Control Signature classes are all located under:
    dax/cont/arg/Topology.h

  The qualifiers (In,Out) and the Domains (Cells,Points) are located under:
    dax/cont/sig/Tag.h
  */
  typedef void ControlSignature( Topology, Topology(Out) );

  /*
  Execution Signature

  In the execution side we now see the first occurance of qualify what information
  from the topology we want. In the base case of the Topology Keyword we only
  receive the TYPE OF CELL the topology contains. This is the most common
  use case of the Topology keyword is to help Field(Point) arguments determine
  the number of points per cell they require.

  But in this use case we actually want the vertices's of the cell. For this
  to happen we wrap the parameter in the Vertices() function which will tell
  the execution environment we want CellVertices.

  I should make it clear that the ExecutionSignature isn't required
  to use every item from the ControlSignature only once. It can use an item
  from the control signature zero or N times. The following  is a valid
  ExecutionSignature:
    typedef void ExecutionSignature( _1, Vertices(_2), Vertices(_1), _2 );

  but please never write that something like that.

  The qualifier Vertices is located at:
    dax/cont/arg/Topology.h

  */
  typedef void ExecutionSignature( Vertices(_1), Vertices(_2) );

  /*
    Execution invocation of worklet

    We have templated based on both the input and output cell types.
    In theory we could have templated off of only the input Cell tag
    and have the output cell tag replaced by using CellTraits<>::CanocialCell.

  */
  template<typename InputCellTag, typename OutputCellTag>
  DAX_EXEC_EXPORT
  void operator()(const dax::exec::CellVertices<InputCellTag> &inVertices,
                  dax::exec::CellVertices<OutputCellTag> &outVertices) const
  {
    /*
    The CellVertices are a light weight wrapping around dax::Tuple which
    doesn't allow the use of the assignment operator. This was done to make
    copying from or into a CellVertices class more explicit


    CellVertices are templated on the cell tag, which allows it to
    determine its size at compile time by using dax::CellTraits to determine
    the number of vertices in a cell. This does mean that currently Dax
    has problems with supporting polyhedrons, and arbitrary polygons.
    */
    outVertices.SetFromTuple(inVertices.GetAsTuple());
  }
};

}

namespace
{

template<class GridType ,class T>
void Dax_Threshold(GridType grid,
                   std::vector<T> &array,
                   T minValue, T  maxValue)
{

  /*
  The basic premise is The dax::cont::Scheduler job is to prepare all the
  required data that is going to be needed in the execution environment to be
  ready for transfer. It can create new Fields, or determine that we should only
  execute on a subset of the passed in data. Once the worklet is running the
  scheduler has no control over what happens.
  It is merely pre and post worklet execution work.

  Scheduler is templated on the device adapter you want the algorithm to run
  on, this device adapter should be equal to what each array handle is
  templated on.
  */
  dax::cont::Scheduler<> scheduler;


  /*
  Take a pre allocated chunk of control/host side memory and tell dax
  that you will want to have access to it on the execution side
  (aka inside worklets )
  */
  dax::cont::ArrayHandle<T> arrayHandle = dax::cont::make_ArrayHandle(array);

  /*
  Construct an array handle with no control side memory allocated. The
  classification worklet will allocate execution side memory, which will not
  transfer that memory to control/host unless explicitly asked too ( by asking
  for a control side portal)
  */
  dax::cont::ArrayHandle< dax::Id > classification;


  /*
  Execute the first step of the threshold algorithm

  Here are the steps that Dax takes to launch the ThresholdClassify worklet

  1. Scheduler uses dax/cont/scheduling/DetermineScheduler.h to determine
     what actual scheduling implementation it should use. Since
     ThresholdClassify is of type dax::exec::WorkletMapCell it
     knows that it should use dax/cont/scheduling/SchedulerCells.h

     This is all implemented using tag based dispatching to a class called
     dax::cont::scheduling::Scheduler, where each worklet type implements
     a specialization given a tag that represents the worklet type


  2. We pass to SchedulerCells the worklet and all the other parameters
     passed to the Invoke method. Invoke can in theory handle an arbitrary number
     of parameters.

     A. Scheduler Cells uses dax::cont::scheduling::VerifyUserArgLength to
        make sure at compile time that the number of items passed to the
        Invoke method matches the number of parameters in the worklets
        ControlSignature. It will state specifically if you have too many or not
        enough items.

     B. Create the concept map binding between the user provided arugments
        and the ControlSignature requirements for each argument.

        We construct a dax::cont::internal::binding<InvocationSignature>
        object around all the user arguments that have been passed in.
        The bindings will iterate over each argument and try to find a valid
        concept that match the user type with the ControlSignature requested
        type (Field, Topology, etc). At this point we have only constructed
        the required dax::cont::arg::ConceptMap specialization for each user
        argument. See the Control Binding section, for more documentation on
        what happens during this process.

        This binding object is what we will wrap pass to Step E as the arguments
        it should use in the execution environment.

        You can read more about how the resolution of ConceptMaps happen
        by reading "Understanding the Dax Scheduler" mainly the Control Binding
        section
          https://gist.github.com/robertmaynard/6038997 (markdown format)
          http://www.daxtoolkit.org/index.php/Understanding_Schedule


      C. Scheduler Cells invokes dax::cont::scheduling::CollectCount with the
        given domain ( Cells in this case ) to determine the number of iterations
        the worklet should be scheduled. The Domain of a worklet details
        what parameters we should trust for length. For example if our domain
        is cell's we should only trust grid class and arrays whose concept for
        that position state it is cell field to get the length.

     D. We allocate execution memory given the size from Step C.
        If an array ControlSignature marks it as Input it is presumed to have
        already been allocated.


     E. We wrap the users worklet inside a dax::exec::internal::Functor
        with the bindings created in Step B. This provides a common
        interface to all worklets. You can consider the bindings classes
        that functor has been given to work similar as a zip iterator plus
        meta data info about each values type to allow automatic conversion
        basic to the correct type, which the dax::exec infrastructure will use
        to construct the proper exec objects for the worklet.


    3.  We pass the exec::Functor over to the DeviceAdapter that the scheduler
        is templated on. Specifically we pass it to the Schedule method
        of the DeviceAdapter. Each DeviceAdapter will have two Schedule
        methods. One accepts a dax::Id which is the number of instances
        to execute in parallel. The second takes in a dax::Id3 which is
        considered to be the max values for a triple nested for loop which
        to run the worklet on.

          DeviceAdapter::Schedule( Functor f, dax::Id size ) is roughly equal to:
            for(dax::Id i=0; i < size; ++i )
              {
              f( dax::Id(i) );
              }

          DeviceAdapter::Schedule( Functor f, dax::Id3 ijk ) is roughly equal to:
            for(dax::Id i=0; i < ijk[0]; ++i )
              for(dax::Id j=0; i < ijk[1]; ++j )
                for(dax::Id k=0; i < ijk[2]; ++k )
                  {
                  f( dax::Id3(i,j,k) );
                  }

        In both cases the users worklet doesn't need to know what scheduling
        algorithm was used to execute the worklet, as that is contained
        in the execution environment infrastructure that calls the users worklet.


        Dax Device Adapters are located in the following locations:
          Serial - dax/cont/internal/DeviceAdapterSerial.h
          TBB - dax/tbb/cont/internal/DeviceAdapterTBB.h
          CUDA - dax/cuda/cont/internal/DeviceAdapterTBB.h
          OpenMP - dax/openmp/cont/internal/DeviceAdapterTBB.h

        The CUDA and OpenMP both forward to thrust specific device adapters.
          Thrust Device Adapter - dax/thrust/cont/internal/DeviceAdapterAlgorithmThrust.h

  4. For each iteration the device adapter will call dax::exec::internal::Functor.
     This class controls the conversion of the iteration index into arguments
     for the actual worklet.

     We create a temporary instance of all the ConceptMap that we have created
     in Step 2B. This is required because on shared memory backends the
     instance of Functor is shared and we don't want two threads
     using the same ConceptMap or they will write / read from the wrong memory
     positions.

     Next we will go through all the ConceptMaps that have been created
     and call the GetExecArg() method of each storing the created object.

     Execution binding is composed of two parts. The execution argument class like
     FieldPortal or FieldConstant which I will call ExecArg, and the binding class
     like BindDirect or BindCellPoints which describes how the ExecArg will be
     used by the worklet.

     The most basic binding is BindDirect which means that for each iteration
     of the functor we will query the ExecArg with the index we are currently on.
     For a more complicated binding like BindCellPoints we have have to query
     he ExecArg for each point of the given cell and return a container class
     ( dax::exec::CellField ) that holds all the field values for the points.

     The issue with BindCellPoints is that it was only given the point field and
     have no information on the topology that we are currently iterating.
     We solve this problem by passing in the entire signature for all the
     worklets parameters as a template signature, and when we construct the
     Binding object we extract the Topology argument which will tell BindCellPoints
     how many points it should allocate for.

     Here are some examples of what GetExecArg() will return for different
     combinations of a ControlSignature and user parameter

       ControlSignatuire + user param = binding + field
        Field(In|Out) + ArrayHandle == dax::exec::arg::BindDirect + dax::exec::arg::FieldPortal
        Field(In|Out) + int ==  dax::exec::arg::BindDirect +  dax::exec::arg::FieldConstant
        Field(Point|In) + ArrayHandle == dax::exec::arg::BindCellPoints + dax::exec::arg::FieldPortal
        Field(Point|In) + int == dax::exec::arg::BindCellPoints + dax::exec::arg::FieldConstant


     We call each ExecArg operator() with the given value we had been passed.
     This can be a flat index (Id) or a 3d range index. It will return an
     execution object that will be passed directly to the worklet.

    Here are some examples of what operator() will return,

    ControlSignatuire + user param = worklet value
      Field(In|Out) + ArrayHandle<float> == float
      Field(In|Out) + int ==  int
      Field(Point|In) + ArrayHandle<float> == dax::exec::CellField<float, CellType>
      Field(Point|In) + int == dax::exec::CellField<int, CellType>


    We finally execute the worklet with the values that had been returned by
    the operator(). We iterate over the ExecArgs calling the SaveExecutionResult()
    method on each. Since some ExecArg return reference objects, they have local
    storage that needs to be written back into global memory
  */
  scheduler.Invoke(worklet::ThresholdClassify< dax::Scalar >(minValue,maxValue),
                  grid, arrayHandle, classification);



  /*
  now determine the type of the output unstructured grid type
  we do this by finding the canonical cell type of the input grid type.
  Usually this is the exact same type as the input cell type, except
  when converting a uniform grid to an unstructured grid
  */
  typedef typename dax::CellTraits<typename GridType::CellTag>::CanonicalCellTag OutCellType;
  typedef dax::cont::UnstructuredGrid<OutCellType> OutGridType;

  /*
    Construct a unstructured grid to store the output of the threshold operation.
    Since we haven't passed it an array handle for points or cells it won't
    allocate any memory on the control/host side unless explicitly asked.
  */
  OutGridType out_grid;

  /*
  Construct a helper class to hold state information about the second
  step of threshold. The state information that we can hold are:
    1. Resolve duplicate topology. By default this is enabled
        SetRemoveDuplicatePoints( bool )

    2. Release the classification information. By default this is enabled
        SetReleaseClassification( bool )
       When enabled we release the classification information as soon as
       possible, which means before running the second worklet but after
       computing the InclusiveScan and UpperBounds on the classification array

    3. Apply the threshold operation on other point fields.
        CompactPointFiled( original_point_field, output_point_field )

  */
  typedef dax::cont::GenerateTopology< worklet::ThresholdTopology > ScheduleGT;
  ScheduleGT generateWorklet(classification);

  /*
  Execute the second step of the threshold algorithm

  Here are the steps that Dax takes to launch the dax::cont::GenerateTopology worklet

  1. Scheduler uses dax/cont/scheduling/DetermineScheduler.h to determine
     what actual scheduling implementation it should use. Since
     dax::cont::GenerateTopology is of type  dax::cont::GenerateTopology it
     knows that it should use dax/cont/scheduling/SchedulerGenerateToplogy.h

     This is all implemented using tag based dispatching to a class called
     dax::cont::scheduling::Scheduler, where each worklet type implements
     a specialization given a tag that represents the worklet type


  2. We pass to SchedulerGenerateToplogy the worklet and all the other parameters
     passed to the Invoke method. Invoke can in theory handle an arbitrary number
     of parameters.

     A. Scheduler Cells uses dax::cont::scheduling::VerifyUserArgLength to
        make sure at compile time that the number of items passed to the
        Invoke method matches the number of parameters in the worklets
        ControlSignature. It will state specifically if you have too many or not
        enough items.

     B. We call DeviceAdapter::InclusiveScan on the classification array passed
        to the GenerateTopology helper

     C. Release the classification array if the GenerateTopology has that
        option enabled

     D. Call DeviceAdapter::UpperBounds on the output of Step 2B.
        Release the inclusive scan ouput

     E. Compute the VisitIndex if needed. The VisitIndex is used when you
        need to iterate multiple times over the same cell. each time you
        visit that cell you want a integer value stating how many times you
        already have been to that cell. this is needed when doing something
        like tetrahedralization

     F. Invoke the default scheduler with a permutation on the grid. This
        worklets only goal is to create the new topology for the output grid.
        By topology I mean something very similar to the vtkCellArray entry
        for a single cell ( all the point indicies it uses. )

       F i) See the Classify Worklet documentation, except the domain this
            time is PermutedCells ( Which you should consider equal to
            Cell for for Step C), and Step E we call the basic
            DeviceAdapter::Schedule( dax::Id ) since we aren't working
            on all the cells of a uniform grid.

     G. Next we check if the user wants to remove duplicates. Currently the
        duplicate removal is a two step affair. This allows us to store
        a mask that can be reused when thresholding other point fields, instead
        of having to recompute it each time.

        Note: All the following algorithms like Copy, Sort, Unique are
        implemented as static calls on the DeviceAdapter.
        Dax Device Adapters are located in the following locations:
          Serial - dax/cont/internal/DeviceAdapterSerial.h
          TBB - dax/tbb/cont/internal/DeviceAdapterTBB.h
          CUDA - dax/cuda/cont/internal/DeviceAdapterTBB.h
          OpenMP - dax/openmp/cont/internal/DeviceAdapterTBB.h

        The CUDA and OpenMP both forward to thrust specific device adapters.
          Thrust Device Adapter - dax/thrust/cont/internal/DeviceAdapterAlgorithmThrust.h

        i) The first step is computing a point mask. We construct an array
           of the same size as the number of points in the input grid and
           iterate over the CellConnections ( think vtkCellArray ) ids of the
           output grid writing writing out 1 in the index for point ids we see

           for example:
            number of input points = 5
            CellConnections values = [1,3,4]
            mask = [0,1,0,1,1]

        ii) The second step is to convert that mask and original point coordinates
            into a reduced point coordinate set while keeping the relative ordering
            of the points intact.

            To this we do the following:
              Run Stream Compact on the Points using the Mask which will extract
              the point coordinates that are valid, the result of the stream
              compact becomes the output geometries point coordinates.

              Copy the output grids cell topology to a temporary array
              Sort the ids in the temporary array
              Unique the ids in the temporary array
              LowerBounds on the temporary array writing back into the
              output grids cell topology

              The last step is what transform the original cells in the
              output grid cell cell connections array from old indice values
              to the new reduced values.

          Note: this might not be the fastest point reduction algorithm, and
          we are interested in somebody researching other ways to find the
          reduced set of point ids.



  */
  scheduler.Invoke( generateWorklet,grid, out_grid);

  std::cout << "Input Grid number of cells: " << grid.GetNumberOfCells() << std::endl;
  std::cout << "Output Grid number of cells: " << out_grid.GetNumberOfCells() << std::endl;
};


//helper class so we can test on all grid types
struct TestOnAllGridTypes
{
  template<typename GridType>
  DAX_CONT_EXPORT
  void operator()(const GridType&) const
  {
    //grid size is 4*4*4 cells
    dax::cont::testing::TestGrid<GridType> grid(4);

    std::vector<float> data_store(5*5*5);

    //fill the vector with random numbers
    std::srand(42); //I like this seed :-)
    std::generate(data_store.begin(),data_store.end(),std::rand);

    const float sum = std::accumulate(data_store.begin(),data_store.end(),0.0f);
    const float average = sum / static_cast<float>(data_store.size());
    const float max = *(std::max_element(data_store.begin(),data_store.end()));

    //use the average as the min boundary so we get only a subset
    Dax_Threshold(grid.GetRealGrid(),data_store,average,max);
  }
};

}

int main(int, char**)
{
  //load up a uniform grid and point based array and threshold
  //this is a basic example of using the Threshold
  dax::cont::UniformGrid<> grid;
  grid.SetExtent( dax::Id3(0,0,0), dax::Id3(4,4,4) );

  //use an array which every value will pass
  std::vector<float> data_store(5*5*5,25);
  float min=0, max=100;

  Dax_Threshold(grid,data_store,min,max);


  //next we are going to use the dax testing infastructure to pump this
  //example through every grid structure and cell type.
  //so that we show we can threshold voxels, triangles, wedges, verts, etc

  dax::cont::testing::GridTesting::TryAllGridTypes(TestOnAllGridTypes());
}
