
#include "SimpleMoab.h"
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

  std::string filename(argv[argc-1]);
  smoab::Interface interface(filename);

  moab::EntityHandle rootHandle = interface.getRoot();

  int inputOption=0;
  while(true)
    {
    std::cout << "What would you like to dump: " << std::endl
              << "\t1: Material and Boundary Info" << std::endl
              << "\t2: All entity sets " << std::endl
              << "\t3: List all 3D entities " << std::endl
              << "\t4: List all 2D entities " << std::endl
              << "\t5: Find Root Parents" << std::endl
              << "\t6: Find Detached Entities" << std::endl;



    std::cin >> inputOption;

    moab::Range range, range2, range3;
    size_t sizes[3];
    smoab::Range r;
    switch(inputOption)
      {
      case 1:
        std::cout << "Find all Material Set items" << std::endl;
        r = interface.findEntitiesWithTag(smoab::MaterialTag(), rootHandle);
        interface.printRange(r);

        std::cout << "Find all Dirichlet Set items" << std::endl;
        r = interface.findEntitiesWithTag(smoab::DirichletTag(), rootHandle);
        interface.printRange(r);

        std::cout << "Find all Neumann Set items" << std::endl;
        r = interface.findEntitiesWithTag(smoab::NeumannTag(), rootHandle);
        interface.printRange(r);
        break;
      case 2:
        std::cout << "Entity Sets" << std::endl;
        r = interface.findEntities(rootHandle, moab::MBENTITYSET);
        interface.printRange(r);
        break;
      case 3:
        std::cout << "3D Geom " << std::endl;
        r = interface.findEntitiesWithTag(smoab::GeomTag(3), rootHandle);
        interface.printRange(r);
        break;
      case 4:
        std::cout << "2D Geom " << std::endl;
        r = interface.findEntitiesWithTag(smoab::GeomTag(2), rootHandle);
        interface.printRange(r);
        break;
      case 5:
        std::cout << "All Entity 'Root' Parents" << std::endl;
        r = interface.findEntityRootParents(rootHandle);
        interface.printRange(r);
        break;
      case 6:
        std::cout << "All Detached Entity Sets" << std::endl;
        r = interface.findDetachedEntities(rootHandle);
        interface.printRange(r);
        break;
      default:
        exit(1);
        break;
      };
    }
  return 1;
  }
