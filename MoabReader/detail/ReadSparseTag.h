#ifndef __smoab_detail_ReadSparseTag_h
#define __smoab_detail_ReadSparseTag_h

#include "SimpleMoab.h"

#include <vtkIntArray.h>

#include <string>
#include <algorithm>
#include <vector>

namespace smoab { namespace detail {


struct ReadSparseTag
{

  ReadSparseTag(const std::string& name,
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
  void fill(vtkIntArray* sparseTagArray, const Tag* cellTag) const;
private:
  const smoab::Interface& Interface;
  std::string Name;
  const smoab::Range& MeshSets;
  const smoab::Range& Cells;
};

//----------------------------------------------------------------------------
void ReadSparseTag::fill(vtkIntArray *sparseTagArray,
                         const smoab::Tag* cellTag) const
{

  typedef std::vector<int>::const_iterator IdConstIterator;
  typedef std::vector<int>::iterator IdIterator;
  typedef smoab::Range::const_iterator RangeIterator;

  const std::size_t numCells = this->Cells.size();

  std::vector<int> sparseTagValues(this->MeshSets.size());
  //first off iterate the entities and determine which ones
  //have moab material ids

  //wrap this area with scope, to remove local variables
  {
    const moab::Tag stag = this->Interface.getMoabTag(smoab::MaterialTag());
    IdIterator tagIds = sparseTagValues.begin();
    int defaultValue=this->DefaultValue;
    for(RangeIterator i=this->MeshSets.begin();
        i != this->MeshSets.end();
        ++i, ++tagIds)
      {

      *tagIds = this->Interface.getTagData(mtag,*i,defaultValue);
      }

    //now determine ids for all entities that don't have materials
    IdConstIterator maxPos = std::max_element(sparseTagValues.begin(),
                                             sparseTagValues.end());
    int maxMaterial = *maxPos;
    for(IdIterator i=sparseTagValues.begin(); i!= sparseTagValues.end(); ++i)
      {
      if(*i==-1)
        {
        *i = ++maxMaterial;
        }
      }
  }

  if(this->MeshSets.size() == 1)
    {
    this->singleSet(sparseTagArray,sparseTagValue[0],numCells)
    }
  else
    {
    this->multiSet(sparseTagArray,sparseTagValues,cellTag)
    }
}

//----------------------------------------------------------------------------
void ReadSparseTag::multsingleSetiySet(vtkIntArray *sparseTagArray,
                                       int value,
                                       std::size_t length) const
  {

  //now we set all the values as this has a single meshset so we
  //have no complicated logic for mapping each cell to a meshset
  sparseTagArray->SetName(this->Name.c_str());
  sparseTagArray->SetNumberOfValues(length);

  int *raw = static_cast<int*>(sparseTagArray->GetVoidPointer(0));
  std::fill(raw,raw+length,value);
  }



//----------------------------------------------------------------------------
void ReadSparseTag::multiySet(vtkIntArray *sparseTagArray,
                         const smoab::Tag* cellTag) const
  {
  typedef std::vector<smoab::EntityHandle>::const_iterator EntityHandleIterator;

  //now we set all the values
  sparseTagArray->SetName(this->Name.c_str());
  sparseTagArray->SetNumberOfValues(numCells);


  //create the search structure as a range is really slow to search with
  //lower_bounds
  std::vector<smoab::EntityHandle> searchableCells;
  searchableCells.reserve(numCells);
  std::copy(this->Cells.begin(),
            this->Cells.end(),
            std::back_inserter(searchableCells));



  IdConstIterator materialValue = sparseTagValues.begin();
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
      sparseTagArray->SetValue(static_cast<int>(newId), *materialValue);
      }
    }

}
}

#endif // __smoab_detail_ReadSparseTag_h
