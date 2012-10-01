#ifndef __smoab_DataSetConverter_h
#define __smoab_DataSetConverter_h

#include "SimpleMoab.h"
#include "CellTypeToType.h"
#include "MixedCellConnectivity.h"

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkFieldData.h>
#include <vtkIntArray.h>
#include <vtkIdTypeArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
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


    this->readProperties(cells,grid->GetCellData());

    smoab::Range moabPoints;
    mixConn.moabPoints(moabPoints);

    vtkNew<vtkPoints> newPoints;
    this->addCoordinates(moabPoints,newPoints.GetPointer());
    grid->SetPoints(newPoints.GetPointer());

    this->readProperties(moabPoints,grid->GetPointData());


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
  void readProperties(smoab::Range const& entities,
                           vtkFieldData* field) const
    {
    if(entities.empty()) { return; }

    //so we get all the tags and parse out the sparse and dense tags
    //that we support
    typedef std::vector<moab::Tag>::const_iterator iterator;
    std::vector<moab::Tag> tags;
    this->Moab->tag_get_tags_on_entity(entities.front(),tags);

    this->readDenseTags(tags,entities,field);
    this->readSparseTags(tags,entities,field);
    }

  //----------------------------------------------------------------------------
  void readSparseTags(std::vector<moab::Tag> &tags,
                      smoab::Range const& entities,
                      vtkFieldData* field) const
    {
    typedef std::vector<moab::Tag>::const_iterator iterator;

    for(iterator i=tags.begin();i!=tags.end();++i)
      {
      moab::TagType tagType;
      moab::DataType tagDataType;

      this->Moab->tag_get_type(*i,tagType);
      this->Moab->tag_get_data_type(*i,tagDataType);

      if(tagType != moab::MB_TAG_SPARSE &&
         tagDataType != moab::MB_TYPE_DOUBLE &&
         tagDataType != moab::MB_TYPE_INTEGER)
        {
        //unsupported type, skip to next tag
        continue;
        }

      std::string name;
      this->Moab->tag_get_name(*i,name);
      std::cout << "Sparse tag: " << name << std::endl;
      }

    }

  void readDenseTags(std::vector<moab::Tag> &tags,
                     smoab::Range const& entities,
                     vtkFieldData* field) const
    {
    typedef std::vector<moab::Tag>::const_iterator iterator;

    //strip out the global id tag, as it's name field is invlaid.
    moab::Tag globalIdTag;
    moab::ErrorCode rval =
        this->Moab->tag_get_handle(GLOBAL_ID_TAG_NAME,1,
                                   moab::MB_TYPE_INTEGER,globalIdTag);
    if(rval==moab::MB_SUCCESS)
      {
      std::remove(tags.begin(),tags.end(),globalIdTag);
      }


    for(iterator i=tags.begin();i!=tags.end();++i)
      {
      moab::TagType tagType;
      moab::DataType tagDataType;

      this->Moab->tag_get_type(*i,tagType);
      this->Moab->tag_get_data_type(*i,tagDataType);

      if(tagType != moab::MB_TAG_DENSE &&
         tagDataType != moab::MB_TYPE_DOUBLE &&
         tagDataType != moab::MB_TYPE_INTEGER)
        {
        //unsupported type, skip to next tag
        continue;
        }

      //read the name of the tag
      std::string name;
      this->Moab->tag_get_name(*i,name);

      //read the number of components of the tag
      int numComps = 1;

      this->Moab->tag_get_length(*i,numComps);

      //read the data if it is one of the two types we support
      int size = entities.size();
      if(tagDataType == moab::MB_TYPE_DOUBLE)
        {
        vtkNew<vtkDoubleArray> array;
        array->SetName(name.c_str());
        array->SetNumberOfComponents(numComps);
        array->SetNumberOfTuples(size);

        //read directly into the double array
        this->Moab->tag_get_data(*i,entities,
                                 array->GetVoidPointer(0));
        field->AddArray(array.GetPointer());
        }
      else if(tagDataType == moab::MB_TYPE_INTEGER)
        {
        vtkNew<vtkIntArray> array;
        array->SetName(name.c_str());
        array->SetNumberOfComponents(numComps);
        array->SetNumberOfTuples(size);

        //read directly into the double array
        this->Moab->tag_get_data(*i,entities,
                                 array->GetVoidPointer(0));
        field->AddArray(array.GetPointer());
        }
      else
        {
        }
      }
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
