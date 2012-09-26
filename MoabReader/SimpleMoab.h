
#ifndef _SimpleMoab_h
#define _SimpleMoab_h

#include "moab/Core.hpp"
#include "moab/Interface.hpp"
#include "moab/Range.hpp"

#include <vtkPoints.h>
#include <vtkUnstructuredGrid.h>

#include <iostream>

namespace smoab
{
//make our range equal to moabs range
typedef moab::Range Range;
typedef moab::EntityHandle EntityHandle;
typedef moab::EntityType EntityType;

//forward declare this->Moab for Tag
struct Interface;

struct Tag
{
  const std::string Name_;

  Tag(std::string const& n):Name_(n)
    {
    }
  const char* name() const { return this->Name_.c_str(); }
  moab::DataType virtual dataType() const { return moab::MB_TYPE_INTEGER; }
};

//lightweight structs to wrap set names, so we detected
//incorrect names at compile time
struct MaterialTag : Tag{ MaterialTag():Tag("MATERIAL_SET"){}};
struct DirichletTag : Tag{ DirichletTag():Tag("DIRICHLET_SET"){}};
struct NeumannTag: Tag{ NeumannTag():Tag("NEUMANN_SET"){}};
struct GroupTag: Tag{ GroupTag():Tag("GROUP"){}};

//light weight wrapper on a moab this->Moab that exposes only the reduced class
//that we need
struct Interface
{
  Interface(const std::string &file):
    Moab(new moab::Core())
    {
    this->Moab->load_file(file.c_str());
    }

  ~Interface(){delete this->Moab;}

  moab::Tag getMoabTag(const smoab::Tag& simpleTag) const
    {
    moab::Tag tag;
    this->Moab->tag_get_handle(simpleTag.name(),
                               1,
                               simpleTag.dataType(),
                               tag);
    return tag;
    }

  smoab::EntityHandle getRoot() const { return this->Moab->get_root_set(); }

  smoab::Range findEntities(const smoab::EntityHandle root, moab::EntityType type) const
    {
    smoab::Range result;
    // get all the sets of that type in the mesh
    this->Moab->get_entities_by_type(root, type, result);
    return result;
    }

  //Find all entities with a given tag. We don't use geom as a tag as that
  //isn't a fast operation. Yes finding the intersection of geom entities and
  //a material / boundary tag will be more work, but it is rarely done currently
  //Returns the found group of entities
  smoab::Range findEntitiesWithTag (const smoab::Tag& tag, smoab::EntityHandle root,
                                    moab::EntityType type = moab::MBENTITYSET) const
    {
    smoab::Range result;

    moab::Tag t = this->getMoabTag(tag);

    // get all the entities of that type in the mesh
    this->Moab->get_entities_by_type_and_tag(root, type, &t, NULL, 1,result);
    return result;
    }

  //Find all entities from a given root of a given dimensionality
  smoab::Range findEntitiesWithDimension(const smoab::EntityHandle root, int dimension) const
    {
    smoab::Range result;
    this->Moab->get_entities_by_dimension(root,dimension,result);
    return result;
    }

  //Find all elements in the database that have children and zero parents.
  //this doesn't find
  smoab::Range findEntityRootParents(smoab::EntityHandle const& root) const
    {
    smoab::Range parents;

    typedef moab::Range::iterator iterator;
    moab::Range sets;

    this->Moab->get_entities_by_type(root, moab::MBENTITYSET, sets);
    for(iterator i=sets.begin(); i!=sets.end();++i)
      {
      int numParents=0,numChildren=0;
      this->Moab->num_parent_meshsets(*i,&numParents);
      if(numParents==0)
        {
        this->Moab->num_child_meshsets(*i,&numChildren);
        if(numChildren>0)
          {
          parents.insert(*i);
          }
        }
      }
    return parents;
    }

  //finds entities that have zero children and zero parents
  smoab::Range findDetachedEntities(moab::EntityHandle const& root) const
    {
    smoab::Range detached;

    typedef moab::Range::iterator iterator;
    moab::Range sets;

    this->Moab->get_entities_by_type(root, moab::MBENTITYSET, sets);
    for(iterator i=sets.begin(); i!=sets.end();++i)
      {
      int numParents=0,numChildren=0;
      this->Moab->num_parent_meshsets(*i,&numParents);
      if(numParents==0)
        {
        this->Moab->num_child_meshsets(*i,&numChildren);
        if(numChildren==0)
          {
          detached.insert(*i);
          }
        }
      }
    return detached;
    }

  //find all children of the entity passed in that has multiple parents
  smoab::Range findEntitiesWithMultipleParents(smoab::EntityHandle const& root)
    {
    smoab::Range multipleParents;
    typedef moab::Range::iterator iterator;

    //for all the elements in the range, find all items with multiple parents
    moab::Range children;
    this->Moab->get_child_meshsets(root,children,0);
    for(iterator i=children.begin(); i!=children.end();++i)
      {
      int numParents=0;
      this->Moab->num_parent_meshsets(*i,&numParents);
      if(numParents>1)
        {
        multipleParents.insert(*i);
        }
      }
    return multipleParents;
    }

  //given a range of entity types, add them to an unstructured grid
  //we return a Range object that holds all the point ids that we used
  //which is sorted and only has unique values.
  ///we use entity types so that we can determine vtk cell type :(
  moab::Range addCells(moab::EntityType start, moab::EntityType end,
                       moab::EntityHandle root, vtkUnstructuredGrid* grid) const
    {
    moab::Range pointRange;
    //even though we have a start and end entity type we can use
    //the highly efficent calls, since we know that are of the same dimension


    //ranges are by nature sorted and unque we just have to return the subset
    //of point entity handles we use

    //the problem is that this call only knows a subsection of all the point ids
    //we are going to use

    return pointRange;
    }

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

  //prints all elements in a range objects
  void printRange(smoab::Range const& range)
    {
    typedef Range::iterator iterator;
    for(iterator i=range.begin(); i!=range.end(); ++i)
      {
      std::cout << "entity id: " << *i << std::endl;
      this->Moab->list_entity(*i);
      }
    }

private:
  moab::Interface* Moab;
};

}

#endif
