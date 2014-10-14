//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef slidingWorklet_h
#define slidingWorklet_h

#include <dax/worklet/MarchingCubes.h>
#include <dax/exec/WorkletMapCell.h>

namespace dax {
namespace worklet {

// -----------------------------------------------------------------------------
class SlidingContour : public dax::exec::WorkletMapCell
{
  typedef dax::cont::ArrayHandle< dax::Vector3 >::PortalExecution TriPortalType;

 public:
  typedef void ControlSignature(TopologyIn, FieldPointIn);
  typedef void ExecutionSignature(AsVertices(_1), _2);

  DAX_CONT_EXPORT SlidingContour(dax::Scalar isoValue,
                                 dax::cont::ArrayHandle<dax::Vector3> outTris):
    IsoValue(isoValue),
    TriPortal(outTris.PrepareForOutput( outTris.GetNumberOfValues() ))
    {

    }

template<class CellTag>
  DAX_EXEC_EXPORT void operator()(
      const dax::exec::CellVertices<CellTag>& verts,
      const dax::exec::CellField<dax::Scalar,CellTag> &values) const
  {
    const int voxelClass =
        internal::marchingcubes::GetHexahedronClassification(IsoValue,values);
    const int numFaces = dax::worklet::internal::marchingcubes::NumFaces[voxelClass];

  }
//     using dax::worklet::internal::marchingcubes::TriTable;
//     // These should probably be available through the voxel class
//     const unsigned char cellVertEdges[12][2] ={
//         {0,1}, {1,2}, {3,2}, {0,3},
//         {4,5}, {5,6}, {7,6}, {4,7},
//         {0,4}, {1,5}, {2,6}, {3,7},
//       };

//     const int voxelClass =
//         internal::marchingcubes::GetHexahedronClassification(IsoValue,values);

//     //save the point ids and ratio to interpolate the points of the new cell
//     for (dax::Id outVertIndex = 0;
//          outVertIndex < outCell.NUM_VERTICES;
//          ++outVertIndex)
//       {
//       const unsigned char edge = TriTable[voxelClass][(inputCellVisitIndex*3)+outVertIndex];
//       int vertA;
//       int vertB;
//       if (verts[cellVertEdges[edge][0]] < verts[cellVertEdges[edge][1]])
//         {
//         vertA = cellVertEdges[edge][0];
//         vertB = cellVertEdges[edge][1];
//         }
//       else
//         {
//         vertA = cellVertEdges[edge][1];
//         vertB = cellVertEdges[edge][0];
//         }

//       // Find the weight for linear interpolation
//       const dax::Scalar weight = (IsoValue - values[vertA]) /
//                                 (values[vertB]-values[vertA]);

//       outCell.SetInterpolationPoint(outVertIndex,
//                                     verts[vertA],
//                                     verts[vertB],
//                                     weight);
//       }

private:
  dax::Scalar IsoValue;
  TriPortalType TriPortal;

};

}
}

#endif
