#ifndef __smoab_detail_ReadMaterialTag_h
#define __smoab_detail_ReadMaterialTag_h

#include "SimpleMoab.h"

#include <vtkIntArray.h>

#include <string>
#include <algorithm>
#include <vector>

namespace smoab { namespace detail {


struct ReadMaterialTag
{

  ReadMaterialTag(const std::string& name,
                  const smoab::Range& ents,
                  const smoab::Range& cells,
                  const smoab::Interface& interface):
    Interface(interface),
    Name(name),
    MeshSets(ents),
    Cells(cells)
    {
    }

  //the cellTag describes the tag mapping between the cells
  //and meshsets.
  void fill(vtkIntArray* materialArray, const Tag* cellTag) const;
private:
  const smoab::Interface& Interface;
  std::string Name;
  const smoab::Range& MeshSets;
  const smoab::Range& Cells;
};

//----------------------------------------------------------------------------
void ReadMaterialTag::fill(vtkIntArray *materialArray,
                           const smoab::Tag* cellTag) const
  {
  typedef std::vector<smoab::EntityHandle>::const_iterator EntityHandleIterator;
  typedef std::vector<int>::const_iterator IdConstIterator;
  typedef std::vector<int>::iterator IdIterator;
  typedef smoab::Range::const_iterator RangeIterator;

  const std::size_t numCells = this->Cells.size();

  std::vector<smoab::EntityHandle> searchableCells;
  searchableCells.reserve(numCells);
  std::copy(this->Cells.begin(),
            this->Cells.end(),
            std::back_inserter(searchableCells));

  std::vector<int> materialIds(this->MeshSets.size());
  //first off iterate the entities and determine which ones
  //have moab material ids

  //wrap this area with scope, to remove local variables
  {
    const moab::Tag mtag = this->Interface.getMoabTag(smoab::MaterialTag());
    IdIterator materialIndex = materialIds.begin();
    int defaultValue=-1;
    for(RangeIterator i=this->MeshSets.begin();
        i != this->MeshSets.end();
        ++i, ++materialIndex)
      {

      *materialIndex = this->Interface.getTagData(mtag,*i,defaultValue);
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

  //now we set all the values
  materialArray->SetName(this->Name.c_str());
  materialArray->SetNumberOfValues(numCells);

  IdConstIterator materialValue = materialIds.begin();
  const int dim = cellTag->value(); //only used if isComparable returns true
  for(RangeIterator i=this->MeshSets.begin();
      i!=this->MeshSets.end(); ++i, ++materialValue)
    {
    //this is a time vs memory trade off, I don't want to store
    //the all the cell ids twice over, lets use more time
    smoab::Range entitiesCells;
    if(cellTag->isComparable())
      {entitiesCells = this->Interface.findEntitiesWithDimension(*i,dim);}
    else
      {entitiesCells = this->Interface.findSubMeshEntities(*i);}

    EntityHandleIterator s_begin = searchableCells.begin();
    EntityHandleIterator s_end = searchableCells.end();
    for(RangeIterator j=entitiesCells.begin(); j != entitiesCells.end();++j)
      {
      EntityHandleIterator result = std::lower_bound(s_begin,
                                                     s_end,
                                                     *j);
      std::size_t newId = std::distance(s_begin,result);
      materialArray->SetValue(static_cast<int>(newId), *materialValue);
      }
    }
}

}
}

#endif // __smoab_detail_ReadMaterialTag_h
