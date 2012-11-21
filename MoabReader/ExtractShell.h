#ifndef __smoab_ExtractShell_h
#define __smoab_ExtractShell_h

#include "SimpleMoab.h"
#include "detail/LoadGeometry.h"
#include "detail/ReadMaterialTag.h"

#include <moab/CN.hpp>

#include <vtkUnstructuredGrid.h>
#include <vtkPolyData.h>

#include <algorithm>

namespace smoab{

//we presume that we are extract the shell of a 3d volume, where the 2d
//surface elements that constructed the 3d volume are contained
//in the input parents
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


  std::vector<EntityHandle> getFaceCells(const smoab::EntityHandle &cell);
  //bool linearizeCell();
};


//----------------------------------------------------------------------------
template<typename T>
void ExtractShell::fill(T* grid)
{
  smoab::Range surfaceCells;
  this->findSkin(this->VCells,surfaceCells);

  //save the results into the grid
  const smoab::GeomTag geom2Tag(2);
  detail::LoadGeometry loadGeom(surfaceCells,geom2Tag.value(),this->Interface);
  loadGeom.fill(grid);

  vtkNew<vtkIntArray> materials;
  materials->SetName(this->MaterialName.c_str());
  detail::ReadMaterialTag materialReader(this->MaterialName,
                                         this->Parents,
                                         surfaceCells,
                                         this->Interface);


  //fill the material array using geom2tag to determine how cells
  //maps back to the parents.
  materialReader.fill(materials.GetPointer(),&geom2Tag);
  grid->GetCellData()->AddArray(materials.GetPointer());
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

  smoab::UsageTable table;
  smoab::FaceToCellMapping fcMap;
  for(Iterator i = volumeCells.begin(); i != volumeCells.end(); ++i)
    {
    //the concern is how to I relate the face cells to the current
    //volume cell, looks like i need a new data structure
    std::vector<EntityHandle> faceCells = this->getFaceCells(*i);

    //map faceCellId to cell entity handle and side index
    fcMap.add(faceCells,*i);

    table.incrementUsage(faceCells);
    }

  surfaceCells = table.getSingleUsageIds();

  //we will remove all cells that have multiple usages from the moab database
  //I really don't care if they already existed or not.

  smoab::Range cellsToRemove = table.GetMultipleUsageIds();

  this->Interface.Moab->remove_entities(this->Interface.getRoot(),
                                        cellsToRemove);

  //just have to store this somewhere
  vtkNew<vtkIntArray> faceIdMap;
  fcMap.generateSkinFaceMap(faceIdMap.GetPointer());

  return true;
}

//----------------------------------------------------------------------------
std::vector<EntityHandle> ExtractShell::getFaceCells(const smoab::EntityHandle &cell)
{
  const moab::Interface *moab = this->Interface.Moab;
  const EntityType volumeCellType = moab->type_from_handle(*i);

  //get the connectivity of the volume cell
  std::vector<moab::EntityHandle> vcConn;
  vcConn.reserve(moab::CN::MAX_NODES_PER_ELEMENT);
  moab->get_connectivity(*i,1,vcConn,false); //false so we get High order connn

  //get the actual length this will allow us to determine
  //if we need to linearize the cell or not down the road
  const int num_connectivity = vcConn.size();


  const short numSides = moab::CN::NumSubEntities( type, two2d );
  for (short side = 0; side < numSides; ++side)
    {
    //for each side of the cell we need to find the connectivity.
    //this api for moab is really unpleasent and filled with
    //terms that aren't clear. We use nodeIncices since it looks
    //like it handles high order nodes
    std::vector<EntityHandle> sideCellConn;
    int numSideNodes;
    int cellConnIndices[moab::CN::MAX_NODES_PER_ELEMENT];

    moab::CN::SubEntityNodeIndices(volumeCellType,
                                 num_connectivity,
                                 two2d,
                                 side,
                                 sideCellType,
                                 numSideNodes,
                                 cellConnIndices
                                 );
    for (int sideCellNode =0; sideCellNode< numSideNodes; ++sideCellNode)
      {

      }
  }


}

#endif // __smoab_ExtractShell_h
