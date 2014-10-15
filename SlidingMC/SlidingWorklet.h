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

#include <dax/cont/internal/EdgeInterpolatedGrid.h>
#include <dax/exec/WorkletMapCell.h>
#include <dax/math/VectorAnalysis.h>
#include <dax/worklet/MarchingCubes.h>

#include "tbb/spin_mutex.h"

namespace dax {

struct PaddedVector3
{
 dax::Vector3 Vec;
 int padding[13];
};

namespace worklet {


//hacky global atomic_lock
::tbb::atomic<std::size_t> contour_atomic_lock;

// -----------------------------------------------------------------------------
class SlidingContour : public dax::exec::WorkletMapCell
{
 public:
  typedef void ControlSignature(TopologyIn, FieldPointIn);
  typedef void ExecutionSignature(AsVertices(_1), _2, WorkId);

  DAX_CONT_EXPORT SlidingContour(dax::Scalar isoValue,
                                 std::vector< dax::PointAsEdgeInterpolation >* tris_storage):
    IsoValue(isoValue),
    TriPortal( tris_storage )
    {
    }

template<class CellTag>
  DAX_EXEC_EXPORT void operator()(
      const dax::exec::CellVertices<CellTag>& verts,
      const dax::exec::CellField<dax::Scalar,CellTag> &values,
      dax::Id cellIndex) const
  {
    using dax::worklet::internal::marchingcubes::TriTable;

    const unsigned char cellVertEdges[12][2] ={
        {0,1}, {1,2}, {3,2}, {0,3},
        {4,5}, {5,6}, {7,6}, {4,7},
        {0,4}, {1,5}, {2,6}, {3,7},
      };
    const int voxelClass =
        internal::marchingcubes::GetHexahedronClassification(IsoValue,values);
    const int numFaces = dax::worklet::internal::marchingcubes::NumFaces[voxelClass];

    if(numFaces == 0)
      {
      return;
      }

    int my_write_index = contour_atomic_lock.fetch_and_add( numFaces * 3 );
    for (dax::Id i =0; i < numFaces; ++i)
      {
      for (dax::Id outVertIndex = 0; outVertIndex < 3; ++outVertIndex, ++my_write_index)
        {
        const unsigned char edge = TriTable[voxelClass][(i*3)+outVertIndex];
        dax::Id first_index, second_index;
        if (verts[cellVertEdges[edge][0]] < verts[cellVertEdges[edge][1]])
          {
          first_index = cellVertEdges[edge][0];
          second_index = cellVertEdges[edge][1];
          }
        else
          {
          first_index = cellVertEdges[edge][1];
          second_index = cellVertEdges[edge][0];
          }

        const dax::Scalar weight = (IsoValue - values[first_index]) /
                                   (values[second_index]- values[first_index]);

        const dax::PointAsEdgeInterpolation interpolationInfo( verts[first_index],
                                                               verts[second_index],
                                                               weight);
        (*this->TriPortal)[my_write_index] = interpolationInfo;
        }

      }
  }

private:
  dax::Scalar IsoValue;
  std::vector< dax::PointAsEdgeInterpolation >* TriPortal;
public:
  ::tbb::atomic<std::size_t>* atomic_lock;
  int cache_line_spacing[3];
};

// -----------------------------------------------------------------------------
template<class InPortalType>
struct InterpolateEdgeToPoints
{
  InterpolateEdgeToPoints(const InPortalType &inPortal,
                          std::vector< dax::PointAsEdgeInterpolation >* edges,
                          vtkSmartPointer<vtkFloatArray> all_triangles,
                          std::size_t numCells) :
  Input(inPortal),
  Interpolation(edges),
  OutputOffset(NULL)
  {
    //reserve space to write into
    const std::size_t current_size = all_triangles->GetNumberOfTuples();
    const std::size_t resize_value = current_size + numCells;

    float* fstart = all_triangles->WritePointer(0, resize_value * 3);
    all_triangles->SetNumberOfTuples( resize_value ); //have to do resize first or this will clear array

    dax::Vector3* start = reinterpret_cast<dax::Vector3*>(fstart);
    this->OutputOffset = start + current_size;
  }


  void operator()(dax::Id index) const
  {
    const dax::PointAsEdgeInterpolation& interpolationInfo = (*this->Interpolation)[index];


    typedef typename InPortalType::ValueType InValueType;
    const InValueType first = this->Input.Get(interpolationInfo.EdgeIdFirst);
    const InValueType second = this->Input.Get(interpolationInfo.EdgeIdSecond);

    this->OutputOffset[index] = dax::math::Lerp(first,second,interpolationInfo.Weight);
  }

  void SetErrorMessageBuffer(
      const dax::exec::internal::ErrorMessageBuffer &) {  }

  InPortalType Input;
  std::vector< dax::PointAsEdgeInterpolation >* Interpolation;
  dax::Vector3* OutputOffset;
};

}
}

#endif
