
#ifndef _tags_h
#define _tags_h

#include "moab/Core.hpp"
#include "moab/Interface.hpp"
#include "moab/Range.hpp"

#include <iostream>
#include <memory>

struct Set
{
  const std::string Name_;
  Set(std::string const& n):Name_(n){}
  const char* name() const { return this->Name_.c_str(); }
};

//lightweight structs to wrap set names, so we detected
//incorrect names at compile time
struct MaterialSet : Set{ MaterialSet():Set("MATERIAL_SET"){} };
struct DirichletSet : Set{ DirichletSet():Set("DIRICHLET_SET"){}};
struct NeumannSet: Set{ NeumannSet():Set("NEUMANN_SET"){}};

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
  int rval = interface->load_file(file.c_str());
  return moab::MB_SUCCESS == rval;
}

//print all the tags in a material, dirichlet or neumann set
void printTagsFromSet(const Set& set, const Interface& interface)
{
  moab::Tag mtag;
  moab::Range sets;
  typedef moab::Range::iterator iterator;

  interface->tag_get_handle(set.name(),1,moab::MB_TYPE_INTEGER,mtag);

  int rval, setId;
  for(iterator i=sets.begin(); i!=sets.end();++i)
    {
    moab::EntityHandle subset = *i;
    rval = interface->tag_get_data(mtag, &subset, 1, &setId);
    if (moab::MB_SUCCESS != rval) return;

    moab::Range entities;
    rval = interface-> get_entities_by_handle(subset, entities, true);
    if (moab::MB_SUCCESS != rval) return;

    std::cout << set.name() << " " << setId << " has "
              << entities.size() << " entities:" << std::endl;
    entities.print("   ");
    entities.clear();
    }
}

//print all sets
void printSets(const Interface& interface)
{
  moab::Range sets;
  int rval;

  rval = interface->get_entities_by_type(0, moab::MBENTITYSET, sets);
  if (moab::MB_SUCCESS != rval) return;

  // print the sets
  rval = interface->list_entities(sets);
  if (moab::MB_SUCCESS != rval) return;

  rval = interface->list_entities(NULL, 1);
}

#endif