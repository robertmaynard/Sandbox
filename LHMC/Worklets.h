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
#include <dax/cont/arg/ExecutionObject.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <dax/math/Compare.h>
#include <dax/math/Exp.h>
#include <dax/math/Trig.h>
#include <dax/math/VectorAnalysis.h>

#include <dax/exec/WorkletMapField.h>
#include <dax/exec/WorkletMapCell.h>
#include <dax/worklet/MarchingCubes.h>

#include <iostream>
#include "CoolWarmColorMap.h"

namespace mandle
{

  class SurfaceCoords : public dax::exec::ExecutionObjectBase
  {
    typedef dax::cont::UnstructuredGrid< dax::CellTagTriangle > DataType;
    typedef DataType::PointCoordinatesType CoordType;
    typedef CoordType::PortalConstExecution PortalType;

  public:
    DAX_CONT_EXPORT
    SurfaceCoords( DataType& grid ):
      Coords( grid.GetPointCoordinates().PrepareForInput() )
      {
      }

  DAX_EXEC_EXPORT dax::Vector3 operator[](int idx) const {
      return this->Coords.Get(idx);
      }

  private:
    PortalType Coords;
  };

}

namespace worklet {


//basic implementation of computing the Mandlebulb if a point
//escapes the mandlebulb of the 10th order
class Mandlebulb : public dax::exec::WorkletMapField
{
public:
  typedef void ControlSignature(Field(In), Field(Out));
  typedef _2 ExecutionSignature(_1);

  DAX_EXEC_EXPORT
  dax::Scalar operator()(const dax::Vector3 &inCoordinate) const
  {
    // The fractal is defined as the number of iterations of
    // pos -> pos^N + inCoordiante it takes to escape. We consider anything
    // outside of the radius sqrt(2) to be escaped.
    const dax::Id MAX_ITERATION = 10;

    dax::Vector3 pos = inCoordinate;
    if (dax::math::MagnitudeSquared(pos) > 2) { return 0; }

    //find the iteration we escape on
    for (dax::Id i=1; i < MAX_ITERATION; ++i)
      {
      pos = this->PowerN(pos) + inCoordinate;
      dax::Scalar escape_value = dax::math::MagnitudeSquared(pos);
      if(escape_value > 2) { return i; }
      }
    return MAX_ITERATION;
  }

  DAX_EXEC_EXPORT
  dax::Vector3 PowerN(dax::Vector3 pos) const
  {
    // Compute the 10th power Mandelbulb. This function raises a coordinate
    // to the 10th power based on White and Nylander's formula.
    const int N = 10;

    const dax::Scalar squareR = dax::math::MagnitudeSquared(pos);
    const dax::Scalar t = dax::math::Sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
    const dax::Scalar thetaN = N * dax::math::ATan2( t, pos[2]);
    const dax::Scalar phiN = N * dax::math::ATan2( pos[1], pos[2] );

    const dax::Scalar powR = dax::math::Pow(squareR,0.5*N); //(r^2)^(N/2) == r^N
    return powR *
        dax::make_Vector3(dax::math::Sin(thetaN) * dax::math::Tan(phiN),
                          dax::math::Sin(thetaN) * dax::math::Sin(phiN),
                          dax::math::Cos(thetaN));
  }
};

class FindLowHigh :  public dax::exec::WorkletMapCell
{
public:
  typedef void ControlSignature(Topology, Field(Point), Field(Out));
  typedef _3 ExecutionSignature(_2);

  template<class CellTag>
  DAX_EXEC_EXPORT
  dax::Vector2 operator()(
    const dax::exec::CellField<dax::Scalar,CellTag> &values) const
  {
    dax::Vector2 lh;
    lh[0] = values[0];
    lh[1] = values[1];
    for(int i=1; i < 8; ++i)
      {
      lh[0] = dax::math::Min(values[i],lh[0]);
      lh[1] = dax::math::Max(values[i], lh[1]);
      }
    return lh;
  }
};

// -----------------------------------------------------------------------------
class MarchingCubesHLCount : public dax::exec::WorkletMapCell
{
  //determine the topology type that we need in the exec env
  typedef dax::cont::UniformGrid< >::TopologyStructConstExecution TopologyType;
  TopologyType Topo; //holds the cell connectivity

  typedef dax::cont::ArrayHandle< dax::Scalar >::PortalConstExecution ValuesPortalType;
  ValuesPortalType ValuesPortal;

  dax::Scalar IsoValue;

public:
  typedef void ControlSignature(Topology, Field, Field(Out));
  typedef _3 ExecutionSignature(_2, WorkId);

  DAX_CONT_EXPORT MarchingCubesHLCount(dax::Scalar isoValue,
                                       dax::cont::UniformGrid< > grid,
                                       dax::cont::ArrayHandle< dax::Scalar > values):
    Topo(grid.PrepareForInput()),
    ValuesPortal(values.PrepareForInput()),
    IsoValue(isoValue)
  {
  }

  DAX_EXEC_EXPORT
  dax::Id operator()(const dax::Vector2& low_high, dax::Id cellIndex )const
  {
    dax::Id count = 0;
    if( low_high[1] >= IsoValue && low_high[0] <= IsoValue )
      {
      dax::exec::CellVertices<dax::CellTagVoxel> verts =
                                  this->Topo.GetCellConnections(cellIndex);
      for(int i=0; i < 8; ++i)
        {
        count |= ( (this->ValuesPortal.Get( verts[i] ) > IsoValue) << i );
        }
      count = dax::worklet::internal::marchingcubes::NumFaces[count];
      }

    return count;
  }

};


// -----------------------------------------------------------------------------
class MarchingCubesHLClip : public dax::exec::WorkletMapField
{
  //determine the topology type that we need in the exec env
  typedef dax::cont::UniformGrid< >::TopologyStructConstExecution TopologyType;
  TopologyType Topology; //holds the cell connectivity

  typedef dax::cont::ArrayHandle< dax::Scalar >::PortalConstExecution ValuesPortalType;
  ValuesPortalType ValuesPortal;

  typedef dax::cont::UniformGrid< >::PointCoordinatesType::PortalConstExecution CoordsPortalType;
  CoordsPortalType CoordsPortal;

  dax::Vector3 Origin;
  dax::Vector3 Location;
  dax::Vector3 Normal;
  dax::Scalar IsoValue;

public:
  typedef void ControlSignature(Field, Field(Out));
  typedef _2 ExecutionSignature(_1, WorkId);

  DAX_CONT_EXPORT MarchingCubesHLClip(dax::Vector3 origin,
                                      dax::Vector3 location,
                                      dax::Vector3 normal,
                                      dax::Scalar isoValue,
                                      dax::cont::UniformGrid< > grid,
                                      dax::cont::ArrayHandle< dax::Scalar > values,
                                      dax::cont::UniformGrid< >::PointCoordinatesType coords)
   :
    Topology(grid.PrepareForInput()),
    ValuesPortal(values.PrepareForInput()),
    CoordsPortal(coords.PrepareForInput()),
    IsoValue(isoValue),
    Origin(origin),
    Location(location),
    Normal(normal)
  {
  }

  DAX_EXEC_EXPORT
  dax::Id operator()(const dax::Vector2& low_high, dax::Id cellIndex )const
  {
    dax::Id count = 0;
    int voxelClass = 0;
    if( low_high[1] >= IsoValue && low_high[0] <= IsoValue )
      {
      const dax::Scalar local_value = dax::dot(Normal,Location);

      dax::exec::CellVertices<dax::CellTagVoxel> verts =
                                  this->Topology.GetCellConnections(cellIndex);

      count = (this->ValuesPortal.Get( verts[0] ) > IsoValue) << 0;
      voxelClass = dax::dot(Normal, CoordsPortal.Get( verts[0] ) - Origin ) > local_value;
      for(int i=1; i < 8; ++i)
        {
        count |= (this->ValuesPortal.Get( verts[i] ) > IsoValue) << i;
        voxelClass |= dax::dot(Normal, CoordsPortal.Get( verts[i] ) - Origin ) > local_value;
        }
      count = dax::worklet::internal::marchingcubes::NumFaces[count];
      }

    if(voxelClass==0) { return 0;}
    else{ return count; }
  }
};

//basic implementation of computing color and norms for triangles
//since dax doesn't have a per vert worklet we are going to replicate
//that worklet type by using worklet map field
class ColorsAndNorms : public dax::exec::WorkletMapField
{
public:
  typedef void ControlSignature(Field(In),ExecObject(), Field(Out), Field(Out));
  typedef void ExecutionSignature(_1, _2, _3, _4);

  DAX_EXEC_EXPORT void operator()( dax::Id vert_index,
                                   const mandle::SurfaceCoords& coords,
                                   dax::Vector3& norm,
                                   dax::Tuple<unsigned char,4>& color  ) const
  {
  //compute normals
  const dax::Id cell_index = 3 * (vert_index / 3);
  const dax::Vector3 first_coord = coords[cell_index];

  const  dax::Vector3 a = coords[cell_index+1] - first_coord;
  const  dax::Vector3 b = coords[cell_index+2] - first_coord;
  norm = dax::math::Normal( dax::math::Cross(a,b) );

  //compute color field, wrap around in both directions
  //with an expanding color field from zero to 1.0
  const dax::Scalar s = dax::math::Abs( dax::dot(coords[vert_index],
                                          dax::make_Vector3(0.09,0.09,0.9)));
  const mandle::CoolWarmColorMap::ColorType &c = this->ColorMap.GetColor(s);
  color[0] = c[0];
  color[1] = c[1];
  color[2] = c[2];
  color[3] = 255;
  }

private:
  mandle::CoolWarmColorMap ColorMap;
};

}
