
#ifndef _SimpleMoab_h
#define _SimpleMoab_h

#include "moab/Core.hpp"
#include "moab/this->Moab.hpp"
#include "moab/Range.hpp"

#include <iostream>

namespace smoab
{
//make our range equal to moabs range
typedef moab::Range Range;
typedef moab::EntityHandle EntityHandle;

//forward declare this->Moab for Tag
struct Interface;

struct Tag
{
  const std::string Name_;
  const int Value_;
  moab::Tag MTag_;

  Tag(Interface const& interface, std::string const& n):Name_(n),Value_(-1)
    {
    this->Moab->tag_get_handle(n.c_str(),1,moab::MB_TYPE_INTEGER,this->MTag_);
    }

  Tag(Interface const& interface, std::string const& n, int v):Name_(n),Value_(v)
    {
    this->Moab->tag_get_handle(n.c_str(),1,moab::MB_TYPE_INTEGER,this->MTag_);
    }

  const char* name() const { return this->Name_.c_str(); }

  bool tagValuesMatch(int otherTagValue) const
   {
   return this->Value_ < 0 || (this->Value_ >= 0 && otherTagValue == Value_);
   }

  moab::Tag moabTag() const { return this->MTag_; }
};

//lightweight structs to wrap set names, so we detected
//incorrect names at compile time
struct MaterialTag : Tag{ MaterialTag(smoab::Interface const& interface):Tag(interface,"MATERIAL_SET"){}};
struct DirichletTag : Tag{ DirichletTag(smoab::Interface const& interface):Tag(interface,"DIRICHLET_SET"){}};
struct NeumannTag: Tag{ NeumannTag(smoab::Interface const& interface):Tag(interface,"NEUMANN_SET"){}};
struct GroupTag: Tag{ GroupTag(smoab::Interface const& interface):Tag(interface,"GROUP"){}};
struct GeomTag: Tag{ GeomTag(smoab::Interface const& interface, int dim):Tag(interface,"GEOM_DIMENSION",dim){}};


//light weight wrapper on a moab this->Moab that exposes only the reduced class
//that we need
struct Interface
{
  Interface(const std::string &file):
    Moab(new moab::Core())
    {
    this->Moab->load_file(file.c_str());
    }

  ~Interface(){delete interface;}


  smoab::EntityHandle getRoot() const { return this->Moab->get_root_set(); }

  smoab::Range findEntities(const smoab::EntityHandle root, moab::EntityType type) const
    {
    smoab::Range result;
    // get all the sets of that type in the mesh
    this->Moab->get_entities_by_type(root, type, result);
    return result;
    }

  //Find all entities with a given tag. If the tag has the optional value
  //like GeomTag we will only match entities that have that value too.
  //Returns the found group of entities
  smoab::Range findEntitiesWithTag (const smoab::Tag& tag, smoab::EntityHandle root,
                                    moab::EntityType type = moab::MBENTITYSET) const
    {
    smoab::Range result;
    moab::Range beforeTagMatches;

    moab::Tag t = tag.moabTag();
    // get all the sets of that type in the mesh
    this->Moab->get_entities_by_type_and_tag(root, type, &t, NULL, 1,
                                             beforeTagMatches);

    //now we have to remove any that doesn't match the tag value
    typedef moab::Range::iterator iterator;
    for(iterator i=beforeTagMatches.begin();
        i != beforeTagMatches.end();
        ++i)
      {
      int tagValue=0;
      moab::EntityHandle handle = *i;
      this->Moab->tag_get_data(tag.moabTag(), &handle, 1, &tagValue);
      if(tag.tagValuesMatch(tagValue))
        {
        result.insert(*i);
        }
      }

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
