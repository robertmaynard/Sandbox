
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

  std::cout << "Find all Material Set items" << std::endl;
  printTagsFromSet(MaterialSet(),interface);

  std::cout << "Find all Dirichlet Set items" << std::endl;
  printTagsFromSet(DirichletSet(),interface);

  std::cout << "Find all Neumann Set items" << std::endl;
  printTagsFromSet(NeumannSet(),interface);

  return 1;
}
