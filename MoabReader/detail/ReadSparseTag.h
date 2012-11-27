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

  ReadSparseTag(const smoab::Range& ents,
                const smoab::Range& cells,
                const smoab::Interface& interface,
                int defaultValue=-1):
    Interface(interface),
    MeshSets(ents),
    Cells(cells),
    DefaultValue(defaultValue)
    {
    }

  //the cellTag describes the tag mapping between the cells
  //and meshsets.
  void fill(vtkIntArray* sparseTagArray, const Tag* cellTag) const;
private:
  void singleSetRead(vtkIntArray *sparseTagArray,
                     const std::vector<int>& values,
                     std::size_t length) const;

  void multiSetRead(vtkIntArray *sparseTagArray,
                    const std::vector<int>& values,
                    std::size_t length,
                    const smoab::Tag* cellTag) const;

  const smoab::Interface& Interface;
  const smoab::Range& MeshSets;
  const smoab::Range& Cells;
  int DefaultValue;
};

//----------------------------------------------------------------------------
void ReadSparseTag::fill(vtkIntArray *sparseTagArray,
                         const smoab::Tag* sparseTag) const
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
    const moab::Tag stag = this->Interface.getMoabTag(*sparseTag);
    IdIterator tagIds = sparseTagValues.begin();
    int defaultValue=this->DefaultValue;
    for(RangeIterator i=this->MeshSets.begin();
        i != this->MeshSets.end();
        ++i, ++tagIds)
      {

      *tagIds = this->Interface.getTagData(stag,*i,defaultValue);
      }

    //now determine ids for all entities that don't have materials
    IdConstIterator maxPos = std::max_element(sparseTagValues.begin(),
                                             sparseTagValues.end());
    int maxValue = *maxPos;
    for(IdIterator i=sparseTagValues.begin(); i!= sparseTagValues.end(); ++i)
      {
      if(*i==-1)
        {
        *i = ++maxValue;
        }
      }
  }

  sparseTagArray->SetName(sparseTag->name());
  sparseTagArray->SetNumberOfValues(numCells);
  if(this->MeshSets.size() == 1)
    {
    this->singleSetRead(sparseTagArray,sparseTagValues,numCells);
    }
  else
    {
    this->multiSetRead(sparseTagArray,sparseTagValues,numCells,sparseTag);
    }
}

//----------------------------------------------------------------------------
void ReadSparseTag::singleSetRead(vtkIntArray *sparseTagArray,
                                  const std::vector<int>& values,
                                  std::size_t length) const
  {

  //now we set all the values as this has a single meshset so we
  //have no complicated logic for mapping each cell to a meshset
  int *raw = static_cast<int*>(sparseTagArray->GetVoidPointer(0));
  std::fill(raw,raw+length,values[0]);
  }

//----------------------------------------------------------------------------
void ReadSparseTag::multiSetRead(vtkIntArray *sparseTagArray,
                                 const std::vector<int>& sparseTagValues,
                                 std::size_t numCells,
                                 const smoab::Tag* sparseTag) const
  {
  typedef std::vector<smoab::EntityHandle>::const_iterator EntityHandleIterator;
  typedef std::vector<int>::const_iterator IdConstIterator;
  typedef smoab::Range::const_iterator RangeIterator;

  //create the search structure as a range is really slow to search with
  //lower_bounds
  std::vector<smoab::EntityHandle> searchableCells;
  searchableCells.reserve(numCells);
  std::copy(this->Cells.begin(),
            this->Cells.end(),
            std::back_inserter(searchableCells));



  IdConstIterator currentTagValue = sparseTagValues.begin();
  const int dim = sparseTag->value(); //only used if isComparable returns true
  for(RangeIterator i=this->MeshSets.begin();
      i!=this->MeshSets.end(); ++i, ++currentTagValue)
    {
    //this is a time vs memory trade off, I don't want to store
    //the all the cell ids twice over, lets use more time
    smoab::Range entitiesCells;
    if(sparseTag->isComparable())
      {entitiesCells = this->Interface.findEntitiesWithDimension(*i,dim,true);}
    else
      {entitiesCells = this->Interface.findHighestDimensionEntities(*i,true);}

    EntityHandleIterator s_begin = searchableCells.begin();
    EntityHandleIterator s_end = searchableCells.end();
    for(RangeIterator j=entitiesCells.begin(); j != entitiesCells.end();++j)
      {
      EntityHandleIterator result = std::lower_bound(s_begin,
                                                     s_end,
                                                     *j);
      std::size_t newId = std::distance(s_begin,result);
      sparseTagArray->SetValue(static_cast<int>(newId), *currentTagValue);
      }
    }

}

}
}

#endif // __smoab_detail_ReadSparseTag_h
