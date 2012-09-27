#ifndef __smoab_DataSetConverter_h
#define __smoab_DataSetConverter_h

#include "SimpleMoab.h"
#include "CellTypeToType.h"

#include <vtkPoints.h>
#include <vtkNew.h>
#include <vtkUnstructuredGrid.h>

namespace smoab
{

class DataSetConverter
{
  smoab::Interface Interface;
  moab::Interface* Moab;

public:
  DataSetConverter(const smoab::Interface& interface ):
    Interface(interface),
    Moab(interface.Moab)
    {
    }

  //----------------------------------------------------------------------------
  bool fill(const smoab::EntityHandle& entity, vtkUnstructuredGrid* grid) const
    {
    smoab::Range pointRange = this->addCells(entity,grid);

    vtkNew<vtkPoints> newPoints;
    this->addCoordinates(pointRange,newPoints.GetPointer());
    grid->SetPoints(newPoints.GetPointer());

    return true;
    }

  //----------------------------------------------------------------------------
  //given a range of entity types, add them to an unstructured grid
  //we return a Range object that holds all the point ids that we used
  //which is sorted and only has unique values.
  ///we use entity types so that we can determine vtk cell type :(
  moab::Range addCells(moab::EntityHandle root,
                       vtkUnstructuredGrid* grid) const
    {

    moab::Range cells = this->Interface.findEntitiesWithDimension(root,
                          this->Moab->dimension_from_handle(root));

    int count = 0;
    while(count != cells.size())
      {
      EntityHandle* connectivity;
      int numVerts=0, iterationCount=0;
      //use the highly efficent calls, since we know that are of the same dimension
      this->Moab->connect_iterate(cells.begin()+count,
                                  cells.end(),
                                  connectivity,
                                  numVerts,
                                  iterationCount);
      count += iterationCount;
      //if we didn't read anything, break!
      if(iterationCount == 0)
        {
        break;
        }

      //now that we have a collection of cells, it is time to identify the
      //cell type and insert them all
      moab::EntityType type = this->Moab->type_from_handle(connectivity[0]);

      }



    //ranges are by nature sorted and unque we just have to return the subset
    //of point entity handles we use
    moab::Range pointRange;
    return pointRange;
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

    double *x,*y,*z;
    int count=0;
    while(count != pointEntities.size())
      {
      int iterationCount=0;
      this->Moab->coords_iterate(pointEntities.begin()+count,
                               pointEntities.end(),
                               x,y,z,
                               iterationCount);
      count+=iterationCount;

      //copy the elements we found over to the vtkPoints
      for(int i=0; i < iterationCount; ++i, rawPoints+=3)
        {
        rawPoints[i] = x[i];
        rawPoints[i+1] = y[i];
        rawPoints[i+2] = z[i];
        }
      }
    }

};

}
#endif // __smoab_DataSetConverter_h
