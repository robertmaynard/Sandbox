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
#ifndef __dax__benchmarks_Mandlebulb_h
#define __dax__benchmarks_Mandlebulb_h

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <dax/exec/ExecutionObjectBase.h>

#include <dax/opengl/internal/OpenGLHeaders.h>

#include <vector>
#include <iostream>

//helper structs to make it easier to pass data around between functions
namespace mandle
{
  class MandlebulbVolume
  {
  public:
  MandlebulbVolume() { }

  MandlebulbVolume( dax::Vector3 origin,
                    dax::Vector3 spacing,
                    dax::Extent3 extent );

  void compute();
  bool isValidSubGrid(std::size_t index, dax::Scalar value);
  bool releaseExecMemForSubGrid(std::size_t index);

  const dax::cont::UniformGrid< >& subGrid( std::size_t index ) const
    { return SubGrids[index]; }
  const dax::cont::ArrayHandle<dax::Vector2>& subLowHighs( std::size_t index ) const
    { return LowHighs[index]; }
  const dax::cont::ArrayHandle<dax::Scalar>& subEscapes( std::size_t index ) const
    { return EscapeIterations[index]; }

  void ReleaseAllResources()
    {
    PerSliceLowHighs.clear();
    SubGrids.clear();
    LowHighs.clear();
    EscapeIterations.clear();
    }

  std::size_t numSubGrids() const { return SubGrids.size(); }

  dax::Vector3 getOrigin() const { return Origin; }
  dax::Vector3 getSpacing() const { return Spacing; }
  dax::Extent3 getExtent() const { return Extent; }
private:
  dax::Vector3 Origin;
  dax::Vector3 Spacing;
  dax::Extent3 Extent;

  std::vector< dax::Vector2 > PerSliceLowHighs;

  std::vector< dax::cont::UniformGrid< > > SubGrids;
  std::vector< dax::cont::ArrayHandle<dax::Vector2> > LowHighs;
  std::vector< dax::cont::ArrayHandle<dax::Scalar> > EscapeIterations;
  };

  class MandlebulbSurface
  {
  typedef dax::Tuple<unsigned char,4> ColorType;

  public:
  void ReleaseAllResources()
    {
    this->Points.ReleaseResources();
    this->Colors.ReleaseResources();
    this->Norms.ReleaseResources();
    }

  dax::cont::ArrayHandle<dax::Vector3> Points;
  dax::cont::ArrayHandle<dax::Vector3> Norms;
  dax::cont::ArrayHandle<ColorType> Colors;
  };
}

//define functions to compute the mandlebulb info
mandle::MandlebulbVolume computeMandlebulb( dax::Vector3 origin,
                                            dax::Vector3 spacing,
                                            dax::Extent3 extent);

mandle::MandlebulbSurface extractSurface( mandle::MandlebulbVolume& vol,
                                          dax::Scalar iteration );

//cut percent represents the ratio from 0 - 1 that we want the cut
//to be along the axis
mandle::MandlebulbSurface extractCut( mandle::MandlebulbVolume& vol,
                                        dax::Scalar cut_percent,
                                        dax::Scalar iteration );

void bindSurface( mandle::MandlebulbSurface& surface,
                  GLuint& coord,
                  GLuint& color,
                  GLuint& norm );

#endif
