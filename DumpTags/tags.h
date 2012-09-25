
#ifndef _tags_h
#define _tags_h

#include "moab/Core.hpp"
#include "moab/Interface.hpp"
#include "moab/Range.hpp"

#include <iostream>
#include <memory>

struct Tag
{
  const std::string Name_;
  const int Value_;
  Tag(std::string const& n):Name_(n),Value_(-1){}
  Tag(std::string const& n, int v):Name_(n),Value_(v){}
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
};

//lightweight structs to wrap set names, so we detected
//incorrect names at compile time
struct MaterialTag : Tag{ MaterialTag():Tag("MATERIAL_SET"){} };
struct DirichletTag : Tag{ DirichletTag():Tag("DIRICHLET_SET"){}};
struct NeumannTag: Tag{ NeumannTag():Tag("NEUMANN_SET"){}};

struct GeomTag: Tag{ GeomTag(int dim):Tag("GEOM_DIMENSION",dim){}};
struct GroupTag: Tag{ GroupTag():Tag("GROUP"){}};

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


//coreate a moab interface to use
bool LoadMoab(std::string const& file, Interface const& interface)
{
  moab::ErrorCode rval = interface->load_file(file.c_str());
  return moab::MB_SUCCESS == rval;
}

void print_set(moab::Range const& range)
{
  //dump all items in a range
  typedef moab::Range::iterator iterator;
  for(iterator i=sets.begin(); i!=sets.end(); ++i)
    {
    std::cout << "entity id: " << *i << std::endl;
    interface->list_entity(*i);
    }
}

//print all the tags in a material, dirichlet or neumann set
void printEntitiesWithTag(const Tag& set, const Interface& interface, moab::EntityHandle root)
{
  moab::ErrorCode rval;
  moab::Tag mtag;
  moab::Range sets;
  typedef moab::Range::iterator iterator;

  interface->tag_get_handle(set.name(),1,moab::MB_TYPE_INTEGER,mtag);

  // get all the sets of that type in the mesh
  rval = interface->get_entities_by_type_and_tag(root, moab::MBENTITYSET, &mtag,
                                          NULL, 1, sets);
  if (moab::MB_SUCCESS != rval) return;

  int tagValue;
  for(iterator i=sets.begin(); i!=sets.end();++i)
    {
    moab::EntityHandle subset = *i;
    rval = interface->tag_get_data(mtag, &subset, 1, &tagValue);
    if (moab::MB_SUCCESS != rval) continue;


    if(!set.tagValuesMatch(tagValue))
      {
      //skip if we don't match the explicit tag value we are looking for
      continue;
      }
    std::cout << "entity id: " << subset << std::endl;
    interface->list_entity(subset);
    }
}

//an attempt to find all parents in the database that have children
void find_parents(const Interface& interface, moab::EntityHandle const& entity, moab::Range& parents)
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

//print all entities given a root
void printEntitySets(const Interface& interface, moab::EntityHandle root)
{
  moab::Range sets;
  moab::ErrorCode rval;

  rval = interface->get_entities_by_type(root, moab::MBENTITYSET, sets);
  if (moab::MB_SUCCESS != rval) return;

  print_set(sets);
}

void print(const Interface& interface, moab::EntityHandle root)
{
  typedef moab::Range::iterator iterator;

  moab::Range parents;
  find_parents(interface,root,parents);

  for(iterator i=parents.begin(); i!=parents.end(); ++i)
    {
    interface->list_entity(*i);
    }
}

//an attempt to find children with multiple parents of a given tag type
void multiple_parents(const GeomTag& gTag, const Interface& interface)
{
  typedef moab::Range::iterator iterator;

  //go grab all the entities in the file. We are only
  //looking for entities that that have a global id of 0

  moab::EntityHandle rootHandle = interface->get_root_set();
  moab::Range parents;
  find_parents(interface,rootHandle,parents);

  moab::Range multipleParents;

  for(iterator i=parents.begin(); i!=parents.end(); ++i)
    {
    moab::Range geomEntities;
    interface->get_entities_by_dimension(*i,gTag.tagValue(), geomEntities);
    for(iterator j=geomEntities.begin(); j!=geomEntities.end();++j)
      {
      int numParents=0;
      interface->num_parent_meshsets(*j,&numParents);
      if(numParents>0)
        {
        multipleParents.insert(*j);
        }
      }
    }
  std::cout << "number of elements with multiple parents " << multipleParents.size() <<  std::endl;
  if(multipleParents.size() > 0)
    {

    }
}


#endif
