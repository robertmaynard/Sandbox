#ifndef __smoab_ExtractShell_h
#define __smoab_ExtractShell_h

#include "SimpleMoab.h"
#include "detail/LoadGeometry.h"
#include "detail/ReadMaterialTag.h"

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
  smoab::Range SCells;

public:
  ExtractShell(const smoab::Interface& interface, const smoab::Range& parents):
    Interface(interface),
    Parents(parents),
    MaterialName("Material")
    {
    const smoab::GeomTag geom2Tag(2);
    //append all the entities cells together into a single range
    typedef smoab::Range::const_iterator iterator;
    for(iterator i=this->Parents.begin(); i!= this->Parents.end(); ++i)
      {
      //find all 2d cells
      smoab::Range entitiesCells =
          this->Interface.findEntitiesWithDimension(*i,geom2Tag);
      this->SCells.insert(entitiesCells.begin(),entitiesCells.end());
      }

    std::cout << "SCells size" << this->SCells.size() << std::endl;
    }

  //----------------------------------------------------------------------------
  template<typename T>
  bool fill(T* grid)
  {
  //we have all the volume cells, what we need is to compare these
  //volume cells to a collection of surface cells
  const smoab::GeomTag geom2Tag(2);

  //save the results into the grid
  detail::LoadGeometry loadGeom(this->SCells,this->Interface);
  loadGeom.fill(grid);

  vtkNew<vtkIntArray> materials;
  materials->SetName(this->MaterialName.c_str());
  detail::ReadMaterialTag materialReader(this->MaterialName,
                                         this->Parents,
                                         this->SCells,
                                         this->Interface);


  //fill the material array using geom2tag to determine how cells
  //maps back to the parents.
  materialReader.fill(materials.GetPointer(),&geom2Tag);
  grid->GetCellData()->AddArray(materials.GetPointer());


  //to properly label the model face / side information we need to query
  //for all the volume elements and find the 2d adjacencies of the 2d element
  //and 3 elements

  return true;
  }

  void materialIdName(const std::string& name) { this->MaterialName = name; }
};
}

#endif // EXTRACTSHELL_H
