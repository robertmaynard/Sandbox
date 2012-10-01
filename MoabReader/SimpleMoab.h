
#ifndef __smoab_SimpleMoab_h
#define __smoab_SimpleMoab_h

#include "moab/Core.hpp"
#include "moab/Interface.hpp"
#include "moab/Range.hpp"
#include "MBTagConventions.hpp"

#include <iostream>

namespace smoab
{
//make our range equal to moabs range
typedef moab::Range Range;
typedef moab::EntityHandle EntityHandle;
typedef moab::EntityType EntityType;

//bring in range functions
using moab::intersect;
using moab::subtract;
using moab::unite;

//forward declare this->Moab for Tag
struct Interface;

//forward declar the DataSetConverter so it can be a friend of Interface
class DataSetConverter;

class Tag
{
  const std::string Name_;
public:
  Tag(std::string const& n):Name_(n)
    {
    }

  virtual ~Tag()
    {
    }

  const char* name() const { return this->Name_.c_str(); }
  moab::DataType virtual dataType() const { return moab::MB_TYPE_INTEGER; }
  virtual bool isComparable() const { return false; }
  virtual int value() const { return int(); }
};

//lightweight structs to wrap set names, so we detected
//incorrect names at compile time. In the future I expect material and
//boundary conditions to be comparable
class MaterialTag : public Tag{ public: MaterialTag():Tag("MATERIAL_SET"){}};
class DirichletTag : public Tag{ public: DirichletTag():Tag("DIRICHLET_SET"){}};
class NeumannTag: public Tag{public:  NeumannTag():Tag("NEUMANN_SET"){}};
class GroupTag: public Tag{ public: GroupTag():Tag("GROUP"){}};

//geom is the only comparable tag, since it can have a dimension.
class GeomTag: public Tag
  {
  int dim;
public:
  GeomTag(int d):Tag("GEOM_DIMENSION"),dim(d){}
  GeomTag():Tag("GEOM_DIMENSION"), dim(0){}

  bool isComparable() const { return dim > 0; }
  int value() const { return dim; }
  };

//light weight wrapper on a moab this->Moab that exposes only the reduced class
//that we need
class Interface
{
public:
  Interface(const std::string &file)
    {
    this->Moab = new moab::Core();
    this->Moab->load_file(file.c_str());
    }

  ~Interface()
    {
    if(this->Moab)
      {
      delete this->Moab;
      this->Moab = NULL;
      }
    }

  //----------------------------------------------------------------------------
  moab::Tag getMoabTag(const smoab::Tag& simpleTag) const
    {
    moab::Tag tag;
    this->Moab->tag_get_handle(simpleTag.name(),
                               1,
                               simpleTag.dataType(),
                               tag);
    return tag;
    }

  //----------------------------------------------------------------------------
  smoab::EntityHandle getRoot() const { return this->Moab->get_root_set(); }

  //----------------------------------------------------------------------------
  smoab::Range findEntities(const smoab::EntityHandle root, moab::EntityType type) const
    {
    smoab::Range result;
    // get all the sets of that type in the mesh
    this->Moab->get_entities_by_type(root, type, result);
    return result;
    }

  //----------------------------------------------------------------------------
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


    if(tag.isComparable())
      {
      int value=0;
      //now we have to remove any that doesn't match the tag value
      smoab::Range resultMatchingTag;
      typedef moab::Range::iterator iterator;
      for(iterator i=result.begin();
          i != result.end();
          ++i)
        {
        value = 0;
        moab::EntityHandle handle = *i;
        this->Moab->tag_get_data(t, &handle, 1, &value);
        if(value == tag.value())
          {
          resultMatchingTag.insert(*i);
          }
        }

      return resultMatchingTag;
      }
    else
      {
      //we return all the items we found
      return result;
      }
    }

  //----------------------------------------------------------------------------
  //Find all entities from a given root of a given dimensionality
  smoab::Range findEntitiesWithDimension(const smoab::EntityHandle root,
                                         int dimension) const
    {
    typedef smoab::Range::iterator iterator;

    smoab::Range result;
    this->Moab->get_entities_by_dimension(root,dimension,result);


    smoab::Range children;
    this->Moab->get_child_meshsets(root,children,0);
    for(iterator i=children.begin(); i !=children.end();++i)
      {
      this->Moab->get_entities_by_dimension(*i,dimension,result);
      }
    return result;
    }

  //----------------------------------------------------------------------------
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
        if(numChildren>=0)
          {
          parents.insert(*i);
          }
        }
      }
    return parents;
    }

  //----------------------------------------------------------------------------
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

  //----------------------------------------------------------------------------
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

  //----------------------------------------------------------------------------
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

  friend class smoab::DataSetConverter;
private:
  moab::Interface* Moab;
};

}

#endif
