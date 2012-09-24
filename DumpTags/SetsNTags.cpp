#include "moab/Core.hpp"
#include "moab/Interface.hpp"
#include "moab/Range.hpp"

#ifdef USE_MPI
#include "moab_mpi.h"
#endif

#include <iostream>

int main(int argc, char **argv) {
    // get the material set tag handle
  moab::Tag mtag;
  moab::ErrorCode rval;
  const char *tag_nms[] = {"MATERIAL_SET", "DIRICHLET_SET", "NEUMANN_SET"};
  moab::Range sets, set_ents;

    // instantiate & load a file
  moab::Interface *mb = new moab::Core();
  const char *par_opt = "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARTITION_DISTRIBUTE;PARALLEL_RESOLVE_SHARED_ENTS;SETS=SETS";

  bool parallel = false;
  if (argc > 2 && !strcmp(argv[1], "-p")) parallel = true;
  else if (argc == 1) {
    std::cout << "Usage: " << argv[0] << "[-p] <filename>" << std::endl;
    return 0;
  }

  if (parallel) 
    rval = mb->load_file(argv[argc-1], 0, par_opt);
  else
    rval = mb->load_file(argv[argc-1]);
  if (moab::MB_SUCCESS != rval) return 1;

    // loop over set types
  for (int i = 0; i < 3; i++) {
    rval = mb->tag_get_handle(tag_nms[i], 1, moab::MB_TYPE_INTEGER, mtag);
    if (moab::MB_SUCCESS != rval) return 1;

      // get all the sets of that type in the mesh
    sets.clear();
    rval = mb->get_entities_by_type_and_tag(0, moab::MBENTITYSET, &mtag,
                                            NULL, 1, sets);
    if (moab::MB_SUCCESS != rval) return 1;

      // iterate over each set, getting entities
    moab::Range::iterator set_it;
    for (set_it = sets.begin(); set_it != sets.end(); set_it++)  {
      moab::EntityHandle this_set = *set_it;

        // get the id for this set
      int set_id;
      rval = mb->tag_get_data(mtag, &this_set, 1, &set_id);
      if (moab::MB_SUCCESS != rval) return 1;

        // get the entities in the set, recursively
      rval = mb->get_entities_by_handle(this_set, set_ents, true);
      if (moab::MB_SUCCESS != rval) return 1;

      std::cout << tag_nms[i] << " " << set_id << " has " 
                << set_ents.size() << " entities:" << std::endl;
      set_ents.print("   ");
      set_ents.clear();
    }
  }

    // do the same for all sets
  sets.clear();
  rval = mb->get_entities_by_type(0, moab::MBENTITYSET, sets);
  if (moab::MB_SUCCESS != rval) return 1;

    // print the sets
  rval = mb->list_entities(sets);
  if (moab::MB_SUCCESS != rval) return 1;

  rval = mb->list_entities(NULL, 1);
  
#ifdef USE_MPI
  if (parallel) {
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << std::flush;
    std::cerr << std::flush;
  }
#endif

  delete mb;
}
