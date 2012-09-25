
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

    moab::Range range, range2;
    switch(inputOption)
      {
      case 1:
        std::cout << "Find all Material Set items" << std::endl;
        findEntitiesWithTag(MaterialTag(interface),interface,rootHandle,range);
        printRange(range,interface);

        std::cout << "Find all Dirichlet Set items" << std::endl;
        findEntitiesWithTag(DirichletTag(interface),interface, rootHandle,range);
        printRange(range,interface);

        std::cout << "Find all Neumann Set items" << std::endl;
        findEntitiesWithTag(NeumannTag(interface),interface, rootHandle,range);
        printRange(range,interface);
        break;
      case 2:
        std::cout << "Raw Sets" << std::endl;
        interface->get_entities_by_type(rootHandle,
                                        moab::MBENTITYSET,
                                        range);
        printRange(range,interface);
        break;
      case 3:
        std::cout << "All Groups " << std::endl;
        findEntitiesWithTag(GroupTag(interface),interface, rootHandle, range);
        printRange(range,interface);
        break;
      case 4:
        std::cout << "3D Geom " << std::endl;
        findEntitiesWithTag(GeomTag(interface,3),interface, rootHandle,range);
        printRange(range,interface);
        break;
      case 5:
        std::cout << "2D Geom " << std::endl;
        findEntitiesWithTag(GeomTag(interface,2),interface, rootHandle, range);
        printRange(range,interface);
        break;
      case 6:
        std::cout << "All parents with atleast one child" << std::endl;
        findParents(interface,rootHandle,range);
        printRange(range,interface);
        break;
      case 7:

        findParents(interface,rootHandle,range);
        findEntitiesWithMultipleParents(interface,range,range2);

        findEntitiesWithTag(GeomTag(interface,2),interface, rootHandle, range);
        range = moab::intersect(range,range2);

        std::cout << "find 2d Entities with 2 parents" <<std::endl;
        printRange(range,interface);

        findEntitiesWithTag(GeomTag(interface,3),interface, rootHandle, range);
        range = moab::intersect(range,range2);

        std::cout << "find 3d Entities with 3 parents" <<std::endl;
        printRange(range,interface);

        break;
      default:
        break;
      };
    }
  return 1;
  }
