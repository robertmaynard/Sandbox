#ifndef __smoab_CellTypeToType_h
#define __smoab_CellTypeToType_h

#include "SimpleMoab.h"
#include "vtkCellType.h"

namespace smoab
{

int vtkCellType(moab::EntityType t, int &num_connect)
  {
  int ctype = -1;
  switch (t)
    {
    case moab::MBEDGE:
      if (num_connect == 2) ctype = VTK_LINE;
      else if (num_connect == 3) ctype = VTK_QUADRATIC_EDGE;
      break;
    case moab::MBTRI:
      if (num_connect == 3) ctype = VTK_TRIANGLE;
      else if (num_connect == 6) ctype = VTK_QUADRATIC_TRIANGLE;
      else if (num_connect == 7) ctype = VTK_BIQUADRATIC_TRIANGLE;
      break;
    case moab::MBQUAD:
      if (num_connect == 4) ctype = VTK_QUAD;
      else if (num_connect == 8) ctype = VTK_QUADRATIC_QUAD;
      else if (num_connect == 9) ctype = VTK_BIQUADRATIC_QUAD;
      break;
    case moab::MBPOLYGON:
      if (num_connect == 4) ctype = VTK_POLYGON;
      break;
    case moab::MBTET:
      if (num_connect == 4) ctype = VTK_TETRA;
      else if (num_connect == 10) ctype = VTK_QUADRATIC_TETRA;
      break;
    case moab::MBPYRAMID:
      if (num_connect == 5) ctype = VTK_PYRAMID;
      else if (num_connect == 13) ctype = VTK_QUADRATIC_PYRAMID;
      break;
    case moab::MBPRISM:
      if (num_connect == 6) ctype = VTK_WEDGE;
      else if (num_connect == 15) ctype = VTK_QUADRATIC_WEDGE;
      break;
    case moab::MBHEX:
      if (num_connect == 8) ctype = VTK_HEXAHEDRON;
      else if (num_connect == 20) ctype = VTK_QUADRATIC_HEXAHEDRON;
      else if (num_connect == 21) ctype = VTK_QUADRATIC_HEXAHEDRON, num_connect = 20;
      else if (num_connect == 27) ctype = VTK_TRIQUADRATIC_HEXAHEDRON;
      break;
    default:
      ctype = -1;
      break;
    }
  return ctype;
  }
}

#endif // CELLTYPETOTYPE_H
