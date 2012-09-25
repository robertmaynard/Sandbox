
#ifndef _tags_h
#define _tags_h

#include "moab/Core.hpp"
#include "moab/Interface.hpp"
#include "moab/Range.hpp"

#include <iostream>
#include <memory>

//light weight wrapper on a moab interface to explicitly manage
//memory.
struct Interface
{
  Interface():Interface_(new moab::Core()){}
  ~Interface(){delete this->Interface_;}

  moab::Interface* operator->() const throw() { return this->Interface_; }
private:
  moab::Interface* Interface_;
};


struct Tag
{
  const std::string Name_;
  const int Value_;
  moab::Tag MTag_;

  Tag(Interface const& interface, std::string const& n):Name_(n),Value_(-1)
    {
    interface->tag_get_handle(n.c_str(),1,moab::MB_TYPE_INTEGER,this->MTag_);
    }

  Tag(Interface const& interface, std::string const& n, int v):Name_(n),Value_(v)
    {
    interface->tag_get_handle(n.c_str(),1,moab::MB_TYPE_INTEGER,this->MTag_);
    }

  const char* name() const { return this->Name_.c_str(); }

  bool hasTagValue() const { return Value_ >= 0; }
  bool tagValue() const { return Value_; }
  bool tagValuesMatch(int otherTagValue) const
   {
   if(!hasTagValue())
     {
     return true;
     }
   return (otherTagValue == Value_);
   }

  moab::Tag moabTag() const { return this->MTag_; }
};

//lightweight structs to wrap set names, so we detected
//incorrect names at compile time
struct MaterialTag : Tag{ MaterialTag(Interface const& interface):Tag(interface,"MATERIAL_SET"){}};
struct DirichletTag : Tag{ DirichletTag(Interface const& interface):Tag(interface,"DIRICHLET_SET"){}};
struct NeumannTag: Tag{ NeumannTag(Interface const& interface):Tag(interface,"NEUMANN_SET"){}};
struct GroupTag: Tag{ GroupTag(Interface const& interface):Tag(interface,"GROUP"){}};
struct GeomTag: Tag{ GeomTag(Interface const& interface, int dim):Tag(interface,"GEOM_DIMENSION",dim){}};


//coreate a moab interface to use
bool LoadMoab(std::string const& file, Interface const& interface)
{
  moab::ErrorCode rval = interface->load_file(file.c_str());
  return moab::MB_SUCCESS == rval;
}


//print all elements in a range
void printRange(moab::Range const& range, const Interface& interface)
{
  typedef moab::Range::iterator iterator;
  for(iterator i=range.begin(); i!=range.end(); ++i)
    {
    std::cout << "entity id: " << *i << std::endl;
    interface->list_entity(*i);
    }

  std::cout << "Printed: " << range.size() << std::endl;
}


//print all the entities with a give tag, be it material, boundary condition, geometery dim
void findEntitiesWithTag(const Tag& tag, const Interface& interface, moab::EntityHandle root, moab::Range& entities)
{

  moab::Tag t = tag.moabTag();
  moab::Range beforeTagMatches;
  // get all the sets of that type in the mesh
  interface->get_entities_by_type_and_tag(root, moab::MBENTITYSET, &t,
                                          NULL, 1, beforeTagMatches);

  //now we have to remove any that doesn't match the tag value
  typedef moab::Range::iterator iterator;
  for(iterator i=beforeTagMatches.begin();
      i != beforeTagMatches.end();
      ++i)
    {
    int tagValue=0;
    moab::EntityHandle handle = *i;
    interface->tag_get_data(tag.moabTag(), &handle, 1, &tagValue);
    if(tag.tagValuesMatch(tagValue))
      {
      entities.insert(*i);
      }
    }
}

//an attempt to find all parents in the database that have children
void findParents(const Interface& interface, moab::EntityHandle const& entity, moab::Range& parents)
{
  typedef moab::Range::iterator iterator;
  moab::Range sets;
  moab::ErrorCode rval;

  rval = interface->get_entities_by_type(entity, moab::MBENTITYSET, sets);
  if(moab::MB_SUCCESS != rval || sets.empty()) return;

  for(iterator i=sets.begin(); i!=sets.end();++i)
    {
    int numParents=0,numChildren=0;
    interface->num_parent_meshsets(*i,&numParents);
    if(numParents==0)
      {
      interface->num_child_meshsets(*i,&numChildren);
      if(numChildren>0)
        {
        parents.insert(*i);
        }
      }
    }
}

//an attempt to find children with multiple parents of a given tag type
void findEntitiesWithMultipleParents(const Interface& interface,
                                     moab::Range const& range,
                                     moab::Range& multipleParents)
{
  typedef moab::Range::iterator iterator;

  //for all the elements in the range, find all items with multiple parents

  for(iterator i=range.begin(); i!=range.end(); ++i)
    {
    moab::Range children;
    interface->get_child_meshsets(*i,children,0);

    for(iterator j=children.begin(); j!=children.end();++j)
      {
      int numParents=0;
      interface->num_parent_meshsets(*j,&numParents);
      if(numParents>1)
        {
        multipleParents.insert(*j);
        }
      }
    }
}


#endif
