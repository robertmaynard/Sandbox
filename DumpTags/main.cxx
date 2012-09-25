
#include "tags.h"
#include <memory>
#include <iostream>

int main(int argc, char **argv)
  {
  //read and dump all tags in the file
  if (argc == 1)
    {
    std::cout << "Usage: " << argv[0] << "<filename>" << std::endl;
    return 0;
    }

  Interface interface;
  bool loaded = LoadMoab(argv[argc-1],interface);
  if(!loaded)
    {
    std::cout << "Failed to load the file. " << std::endl;
    return 0;
    }

  moab::EntityHandle rootHandle = interface->get_root_set();

  int inputOption=0;
  while(true)
    {
    std::cout << "What would you like to dump: " << std::endl
              << "\t1: Material and Boundary Info" << std::endl
              << "\t2: All entity sets " << std::endl
              << "\t3: List all Groups " << std::endl
              << "\t4: List all 3D entities " << std::endl
              << "\t5: List all 2D entities " << std::endl
              << "\t6: Find Parents" << std::endl
              << "\t7: Find Entities with multiple parents" << std::endl;


    std::cin >> inputOption;
    //inputOption = 6;

    switch(inputOption)
      {
      case 1:
        std::cout << "Find all Material Set items" << std::endl;
        printEntitiesWithTag(MaterialTag(),interface, rootHandle);

        std::cout << "Find all Dirichlet Set items" << std::endl;
        printEntitiesWithTag(DirichletTag(),interface, rootHandle);

        std::cout << "Find all Neumann Set items" << std::endl;
        printEntitiesWithTag(NeumannTag(),interface, rootHandle);
        break;
      case 2:
        std::cout << "Raw Sets" << std::endl;
        printEntitySets(interface, rootHandle);
        break;
      case 3:
        std::cout << "All Groups " << std::endl;
        printEntitiesWithTag(GroupTag(),interface, rootHandle);
        break;
      case 4:
        std::cout << "3D Geom " << std::endl;
        printEntitiesWithTag(GeomTag(3),interface, rootHandle);
        break;
      case 5:
        std::cout << "2D Geom " << std::endl;
        printEntitiesWithTag(GeomTag(2),interface, rootHandle);
        break;
      case 6:
        std::cout << "All parents with atleast one child" << std::endl;
        moab::Range parents;
        find_parent(interface,rootHandle,parents);
        print_set(parents);
        break;
      case 7:
        std::cout << "find duplicate 2d Entities" <<std::endl;
        multiple_parents(GeomTag(2),interface);
        std::cout << "find duplicate 3d Entities" <<std::endl;
        multiple_parents(GeomTag(3),interface);
        break;
      default:
        break;
      };
    }
  return 1;
  }