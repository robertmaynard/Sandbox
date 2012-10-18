
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
              << "\t6: Find Detached Entities" << std::endl
              << "\t8: Intersect Surfaces and Volume faces" << std::endl;



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
      case 8:
        {
        smoab::Range parents = interface.findEntityRootParents(rootHandle);


        smoab::Range geomEnts = interface.findEntitiesWithTag(smoab::GeomTag(3), rootHandle);
        smoab::Range surfaceEnts = interface.findEntitiesWithTag(smoab::GeomTag(2), rootHandle);
        smoab::Range surfaces = smoab::intersect(surfaceEnts,parents);
        smoab::Range solids = smoab::intersect(geomEnts,parents);

        geomEnts.clear();
        surfaceEnts.clear();

        typedef smoab::Range::const_iterator Iterator;
        std::cout << "Number of subset Ents: " << surfaces.size() << std::endl;
        for(Iterator surface = surfaces.begin();
            surface != surfaces.end();
            ++surface)
        {
          moab::EntityHandle handle = *surface;
          smoab::Range surfaceQuads = interface.findEntities(handle,moab::MBQUAD);
          smoab::Range surfaceVerts = interface.findEntities(handle,moab::MBVERTEX);

          for(Iterator solid=solids.begin();
              solid != solids.end();
              ++solid)
            {
              std::cout << "solid" << std::endl;
            //look for all vertex indices in the hex model.
            //than lets look and see if they are contained in the surface entity
            //sets ever.
            moab::EntityHandle solidHandle = *solid;
            smoab::Range hexes = interface.findEntities(solidHandle,moab::MBHEX);
            for (Iterator hex = hexes.begin();
                 hex != hexes.end(); ++hex)
              {
              smoab::Range quadAdj = interface.findAdjacentEntities(*hex,2);
              smoab::Range intersectionResult = smoab::intersect(quadAdj,surfaceQuads);
              if(intersectionResult.size() == 1)
                {
                smoab::Range vertAdj = interface.findAdjacentEntities(intersectionResult[0],0);
                smoab::Range fff =  smoab::intersect(vertAdj,surfaceVerts);
                if(fff.size() == 4)
                {
                std::cout << "The entity id: " << *hex  << "relates to " << fff.size()  << " verts " << std::endl;
                }
              }
            }




        }





          //we need to look at Adjacencies of the cell as that holds
          //the face and edge!!!!!! when it has a matching surface face
          //aka
          //entity id: 10376293541461622798
          // Global id = 14
          // Adjacencies:
          //  Vertex 80, Vertex 70, Vertex 81, Vertex 86, Vertex 270, Vertex 262, Vertex 271, Vertex 276
          //  Edge 29, Edge 32, Edge 46, Edge 48
          //  Quad 14
          }
        }
        break;
      default:
        exit(1);
        break;
      };
    }
  return 1;
  }
