
#ifndef _SimpleMoab_h
#define _SimpleMoab_h

#include "moab/Core.hpp"
#include "moab/Interface.hpp"
#include "moab/Range.hpp"

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

  //given a range of cells, add them to an object of type GridType. GridType
  //is a template so we don't have to include any vtk headers. Thruthfully we
  //expect it to be an unstructured grid
  template<typename GridType>
  void addCells(moab::EntityType type, smoab::Range cells, GridType* grid) const
    {



    }

  template<typename PointType>
  void addCoordinates(smoab::Range cells, PointType* pointContainer) const
    {

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
