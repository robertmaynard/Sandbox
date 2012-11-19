#ifndef __smoab_DataSetConverter_h
#define __smoab_DataSetConverter_h

#include "SimpleMoab.h"
#include "detail/MixedCellConnectivity.h"

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
  const smoab::Tag *Tag;
  bool ReadMaterialIds;
  bool ReadProperties;
  std::string MaterialName;

public:
  DataSetConverter(const smoab::Interface& interface, const smoab::Tag* tag):
    Interface(interface),
    Moab(interface.Moab),
    Tag(tag),
    ReadMaterialIds(false),
    ReadProperties(false),
    MaterialName("Material")
    {
    }

  void readMaterialIds(bool add) { this->ReadMaterialIds = add; }
  bool readMaterialIds() const { return this->ReadMaterialIds; }

  void materialIdName(const std::string& name) { this->MaterialName = name; }
  const std::string& materialIdName() const { return this->MaterialName; }

  void readProperties(bool readProps) { this->ReadProperties = readProps; }
  bool readProperties() const { return this->ReadProperties; }

  //----------------------------------------------------------------------------
  //given a range of entity handles merge them all into a single unstructured
  //grid. Currently doesn't support reading properties.
  //Will read in material ids,  if no material id is assigned to an entity,
  //its cells will be given an unique id
  bool fill(const smoab::Range& entities,
            vtkUnstructuredGrid* grid) const
    {
    //create a helper datastructure which can determines all the unique point ids
    //and converts moab connecitvity info to vtk connectivity
    moab::Range cells;

    //append all the entities cells together into a single range
    typedef smoab::Range::const_iterator iterator;
    for(iterator i=entities.begin(); i!= entities.end(); ++i)
      {
      if(this->Tag->isComparable())
        {
        //if we are comparable only find the cells that match our tags dimension
        smoab::Range entitiesCells = this->Interface.findEntitiesWithDimension(*i,Tag->value());
        cells.insert(entitiesCells.begin(),entitiesCells.end());
        }
      else
        {
        //this is a bad representation of all other tags, but we are presuming that
        //neuman and dirichlet are on entitysets with no children
        this->Moab->get_entities_by_handle(*i,cells);
        }
      }

    smoab::Range points;
    this->loadCellsAndPoints(cells,points,grid);

    if(this->readMaterialIds())
      {
      typedef std::vector<smoab::EntityHandle>::const_iterator EntityHandleIterator;
      typedef std::vector<int>::const_iterator IdConstIterator;
      typedef std::vector<int>::iterator IdIterator;

      std::vector<smoab::EntityHandle> searchableCells;
      searchableCells.reserve(grid->GetNumberOfCells());
      std::copy(cells.begin(),cells.end(),std::back_inserter(searchableCells));
      cells.clear(); //release memory we don't need


      std::vector<int> materialIds(entities.size());
      //first off iterate the entities and determine which ones
      //have moab material ids

        //wrap this area with scope, to remove local variables
      {
        smoab::MaterialTag tag;
        IdIterator materialIndex = materialIds.begin();
        for(iterator i=entities.begin();
            i != entities.end();
            ++i, ++materialIndex)
          {
          moab::Tag mtag = this->Interface.getMoabTag(tag);

          int value=-1;
          this->Moab->tag_get_data(mtag,&(*i),1,&value);
          *materialIndex=static_cast<int>(value);
          }

        //now determine ids for all entities that don't have materials
        IdConstIterator maxPos = std::max_element(materialIds.begin(),
                                                 materialIds.end());
        int maxMaterial = *maxPos;
        for(IdIterator i=materialIds.begin(); i!= materialIds.end(); ++i)
          {
          if(*i==-1)
            {
            *i = ++maxMaterial;
            }
          }
      }

      //now we create the material field, and set all the values
      vtkNew<vtkIntArray> materialSet;
      materialSet->SetName(this->materialIdName().c_str());
      materialSet->SetNumberOfValues(grid->GetNumberOfCells());

      IdConstIterator materialValue = materialIds.begin();
      for(iterator i=entities.begin(); i!= entities.end(); ++i, ++materialValue)
        {
        //this is a time vs memory trade off, I don't want to store
        //the all the cell ids twice over, lets use more time
        smoab::Range entitiesCells;
        if(this->Tag->isComparable())
          {entitiesCells = this->Interface.findEntitiesWithDimension(*i,Tag->value());}
        else
          {this->Moab->get_entities_by_handle(*i,entitiesCells);}

        EntityHandleIterator s_begin = searchableCells.begin();
        EntityHandleIterator s_end = searchableCells.end();
        for(iterator j=entitiesCells.begin(); j != entitiesCells.end();++j)
          {
          EntityHandleIterator result = std::lower_bound(s_begin,
                                                         s_end,
                                                         *j);
          std::size_t newId = std::distance(s_begin,result);
          materialSet->SetValue(static_cast<int>(newId), *materialValue);
          }
        }

      grid->GetCellData()->AddArray(materialSet.GetPointer());

      }

    return true;
    }

  //----------------------------------------------------------------------------
  //given a single entity handle create a unstructured grid from it.
  //optional third parameter is the material id to use if readMaterialIds
  //is on, and no material sparse tag is found for this entity
  bool fill(const smoab::EntityHandle& entity,
            vtkUnstructuredGrid* grid,
            const int materialId=0) const
    {
    //create a helper datastructure which can determines all the unique point ids
    //and converts moab connecitvity info to vtk connectivity
    moab::Range cells;
    if(this->Tag->isComparable())
      {
      //if we are comparable only find the cells that match our tags dimension
      cells = this->Interface.findEntitiesWithDimension(entity,Tag->value());
      }
    else
      {
      //this is a bad representation of all other tags, but we are presuming that
      //neuman and dirichlet are on entitysets with no children
      this->Moab->get_entities_by_handle(entity,cells);
      }

    smoab::Range points;
    this->loadCellsAndPoints(cells,points,grid);


    if(this->readProperties())
      {
      this->readProperties(cells,grid->GetCellData());
      this->readProperties(points,grid->GetPointData());
      }

    if(this->readMaterialIds())
      {
      this->readSparseTag(smoab::MaterialTag(),entity,
                          grid->GetNumberOfCells(),
                          grid->GetCellData(),
                          materialId);
      }
    return true;
    }

  //----------------------------------------------------------------------------
  void loadCellsAndPoints(const smoab::Range& cells,
                          smoab::Range& points,
                          vtkUnstructuredGrid* grid) const
    {

    smoab::detail::MixedCellConnectivity mixConn(cells,this->Moab);

    //now that mixConn has all the cells properly stored, lets fixup
    //the ids so that they start at zero and keep the same logical ordering
    //as before.
    vtkIdType numCells, connLen;
    mixConn.compactIds(numCells,connLen);
    this->setGridsTopology(mixConn,grid,numCells,connLen);

    mixConn.moabPoints(points);

    vtkNew<vtkPoints> newPoints;
    this->addCoordinates(points,newPoints.GetPointer());
    grid->SetPoints(newPoints.GetPointer());

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
    }

  //----------------------------------------------------------------------------
  bool readSparseTag(smoab::Tag tag,
                      smoab::EntityHandle const& entity,
                      vtkIdType length,
                      vtkFieldData* field,
                      vtkIdType defaultValue) const
    {

    typedef std::vector<moab::Tag>::const_iterator iterator;
    moab::Tag mtag = this->Interface.getMoabTag(tag);

    int value=0;
    moab::ErrorCode rval = this->Moab->tag_get_data(mtag,&entity,1,&value);
    if(rval!=moab::MB_SUCCESS)
      {
      value = defaultValue;
      }

    vtkNew<vtkIntArray> materialSet;
    materialSet->SetNumberOfValues(length);
    materialSet->SetName(this->materialIdName().c_str());

    int *raw = static_cast<int*>(materialSet->GetVoidPointer(0));
    std::fill(raw,raw+length,value);

    field->AddArray(materialSet.GetPointer());

    return true;
    }

  //----------------------------------------------------------------------------
  void readDenseTags(std::vector<moab::Tag> &tags,
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

      //make sure it is only dense
      if(tagType != moab::MB_TAG_DENSE)
        {
        continue;
        }
      //and only integer and double
      if(tagDataType != moab::MB_TYPE_DOUBLE &&
         tagDataType != moab::MB_TYPE_INTEGER)
        {
        //unsupported type, skip to next tag
        continue;
        }

      //read the name of the tag
      std::string name;
      name.reserve(32);
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
  void setGridsTopology(smoab::detail::MixedCellConnectivity const& mixedCells,
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
