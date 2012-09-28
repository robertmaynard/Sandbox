#ifndef __smoab_DataSetConverter_h
#define __smoab_DataSetConverter_h

#include "SimpleMoab.h"
#include "CellTypeToType.h"
#include "MixedCellConnectivity.h"

#include <vtkCellArray.h>
#include <vtkIdTypeArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>

#include <algorithm>

namespace smoab{
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
    smoab::MixedCellConnectivity mixConn(cells,this->Moab);

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
    if(pointEntities.empty()) { return; }


    //so we get all the tags
    typedef std::vector<moab::Tag>::const_iterator iterator;
    std::vector<moab::Tag> tags;
    this->Moab->tag_get_tags_on_entity(pointEntities.front(),tags);

    //foreach tag,
    for(iterator i=tags.begin();i!=tags.end();++i)
      {
      void* tagData;
      this->Moab->tag_get_data(*i,pointEntities,tagData);

      //now we determine type (int/double) and verify it is dense
      //we really need a function that does this all by hand
      }

    }

  //----------------------------------------------------------------------------
  void readCellProperties(smoab::Range const& cellEntities,
                          vtkUnstructuredGrid* grid) const
    {
    //mixed cells can give use all the entities in a vector that are cell ids

    }

  //----------------------------------------------------------------------------
  void fillGrid(smoab::MixedCellConnectivity const& mixedCells,
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
