#ifndef __smoab_ExtractShell_h
#define __smoab_ExtractShell_h

#include "SimpleMoab.h"
#include "detail/LoadGeometry.h"
#include "detail/UsageTable.h"

#include <vtkUnstructuredGrid.h>
#include <vtkPolyData.h>

#include <algorithm>
#include <cassert>

namespace smoab{

/*
This is a work in progress that had to be left as it was not needed.
It might be needed again in the future when Neumann sets are not aviable
*/


class ExtractShell
{
  const smoab::Interface& Interface;
  const smoab::Range& Parents;
  std::string MaterialName;
  smoab::Range VCells;

public:
  ExtractShell(const smoab::Interface& interface, const smoab::Range& parents):
    Interface(interface),
    Parents(parents),
    MaterialName("Material"),
    VCells()
    {
    const smoab::GeomTag geom3Tag(3);
    //append all the entities cells together into a single range
    typedef smoab::Range::const_iterator iterator;
    for(iterator i=this->Parents.begin(); i!= this->Parents.end(); ++i)
      {
      //find all 3d cells
      smoab::Range entitiesCells =
          this->Interface.findEntitiesWithDimension(*i,geom3Tag.value());
      this->VCells.insert(entitiesCells.begin(),entitiesCells.end());
      }
    }

  void materialIdName(const std::string& name) { this->MaterialName = name; }

  template<typename T>
  void fill(T* grid);

private:
  bool findSkin(const smoab::Range &volumeCells,
                smoab::Range &surfaceCells);


  std::vector<EntityHandle> findFaceCells(const smoab::EntityHandle &cell);
  std::vector<int> GetFaceRegionIds(const std::vector<EntityHandle> &faces);
};


//----------------------------------------------------------------------------
template<typename T>
void ExtractShell::fill(T* grid)
{
  //we need to read the volume material ids first, as they help identify
  //which volume cells should be kept
  smoab::Range surfaceCells;
  this->findSkin(this->VCells,surfaceCells);

  //save the results into the grid
  const smoab::GeomTag geom2Tag(2);
  detail::LoadGeometry loadGeom(surfaceCells,geom2Tag.value(),this->Interface);
  loadGeom.fill(grid);
}


//----------------------------------------------------------------------------
bool ExtractShell::findSkin(const smoab::Range &volumeCells,
                            smoab::Range &surfaceCells)
{
  if(volumeCells.empty())
    {
    return false;
    }

  typedef smoab::Range::const_iterator Iterator;

  smoab::detail::UsageTable table;

  this->Interface.createAdjacencies(volumeCells,2);
  for(Iterator i = volumeCells.begin(); i != volumeCells.end(); ++i)
    {
    std::vector<smoab::EntityHandle> faceCells = this->findFaceCells(*i);

    //get the face cell region id's if they exist, we will use
    //-1 to represent no region id for the face
    std::vector<int> faceRegionIds = this->GetFaceRegionIds(faceCells);

    //increment these face cells relationship to the volume cells material
    //id.
    table.incrementUsage(faceCells,faceRegionIds);
    }

  //we will remove all cells that have multiple usages from the moab database
  //I really don't care if they already existed or not.
  smoab::Range cellsToRemove = table.multipleUsage();
  this->Interface.remove(cellsToRemove);

  surfaceCells = table.singleUsage();

  return true;
}

//----------------------------------------------------------------------------
std::vector<smoab::EntityHandle> ExtractShell::findFaceCells(
                                             const smoab::EntityHandle &cell)
{
  const int two2d(2);
  smoab::Range adjacenices = this->Interface.findAdjacencies(cell,two2d);
  std::vector<smoab::EntityHandle> faces =
                             this->Interface.sideElements(cell,two2d);

  smoab::Range check;
  std::copy(faces.rbegin(),faces.rend(),moab::range_inserter(check));
  assert(adjacenices.contains(check));

  return faces;
}

//----------------------------------------------------------------------------
std::vector<int> ExtractShell::GetFaceRegionIds(
                                        const std::vector<EntityHandle> &faces)
{
  //todo properly look up the correct region/material id for each face
  return std::vector<int>(1,faces.size());
}

}

#endif // __smoab_ExtractShell_h
