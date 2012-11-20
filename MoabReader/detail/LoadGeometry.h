#ifndef __smoab_detail_LoadGeometry_h
#define __smoab_detail_LoadGeometry_h

#include "SimpleMoab.h"
#include "MixedCellConnectivity.h"

#include <vtkCellArray.h>
#include <vtkIdTypeArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPointSet.h>
#include <vtkPolyData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>
namespace smoab { namespace detail {

namespace {

//empty definition, only allow specializations to compile
template<typename VTKGridType> struct setCells;

template<> struct setCells<vtkUnstructuredGrid>
{
  typedef vtkUnstructuredGrid VTKGridType;
  void operator()(VTKGridType* grid, int,
                  vtkUnsignedCharArray* types,
                  vtkIdTypeArray* locations,
                  vtkCellArray*   cells) const
    {
    grid->SetCells(types,locations,cells,NULL,NULL);
    }
};

template<> struct setCells<vtkPolyData>
{
  typedef vtkPolyData VTKGridType;
  void operator()(VTKGridType* grid, int dim,
                  vtkUnsignedCharArray*,
                  vtkIdTypeArray*,
                  vtkCellArray*   cells) const
    {
    //this is explicitly checking 0 and 3
    //a dimensionality of zero is returned when we are reading
    //none topology tags so this is a hack, as we presume those
    //are quads/triangles
    if( dim == 0 || dim == 3)
      {
      grid->SetPolys(cells);
      }
    else if( dim == 1 )
      {
      grid->SetVerts(cells);
      }
    else if( dim == 2)
      {
      grid->SetLines(cells);
      }

    }
};

}
//basic class that given a range of moab cells will convert them to am unstructured grid.
//holds only references to input cells, so they can't go out of scope
class LoadGeometry
{
  moab::Interface* Interface;
  const smoab::Tag *TopologyTag;
  smoab::detail::MixedCellConnectivity MixConn;
  smoab::Range Points;

public:
  //----------------------------------------------------------------------------
  //warning cells input to constructor is held by reference
  LoadGeometry(const smoab::Range& cells,
               const smoab::Tag* topologyTag,
               const smoab::Interface& interface):
    Interface(interface.Moab),
    TopologyTag(topologyTag),
    MixConn(cells,interface.Moab),
    Points()
    {
    }

  const smoab::Range& moabPoints() const { return this->Points; }

  //----------------------------------------------------------------------------
  //todo: have support for using only a subsection of the input cells
  template<typename vtkDataSetType>
  void fill(vtkDataSetType *dataSet)
    {
    //now that mixConn has all the cells properly stored, lets fixup
    //the ids so that they start at zero and keep the same logical ordering
    //as before.
    vtkIdType numCells, connLen;
    this->MixConn.compactIds(numCells,connLen);
    this->addGridsTopology(dataSet,numCells,connLen);
    this->addCoordinates(dataSet);
    }

private:
  //----------------------------------------------------------------------------
  void addCoordinates(vtkPointSet *grid)
    {
    //this is sorta of hackish as moabPoints is only valid
    //after compactIds has been called
    this->MixConn.moabPoints(this->Points);

    //since the smoab::range are always unique and sorted
    //we can use the more efficient coords_iterate
    //call in moab, which returns moab internal allocated memory
    vtkNew<vtkPoints> newPoints;
    newPoints->SetDataTypeToDouble();
    newPoints->SetNumberOfPoints(this->Points.size());

    //need a pointer to the allocated vtkPoints memory so that we
    //don't need to use an extra copy and we can bypass all vtk's check
    //on out of bounds
    double *rawPoints = static_cast<double*>(newPoints->GetVoidPointer(0));
    this->Interface->get_coords(this->Points,rawPoints);

    grid->SetPoints(newPoints.GetPointer());
    }

  //----------------------------------------------------------------------------
  template<typename vtkDataSetType>
  void addGridsTopology(vtkDataSetType* grid,
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

    this->MixConn.copyToVtkCellInfo(rawArray,rawLocations,rawTypes);

    vtkNew<vtkCellArray> cells;
    cells->SetCells(numCells,cellArray.GetPointer());


    setCells<vtkDataSetType>()(grid,
                               this->TopologyTag->value(),
                               cellTypes.GetPointer(),
                               cellLocations.GetPointer(),
                               cells.GetPointer());
    }
};


} }

#endif // __smoab_detail_LoadGeometry_h
