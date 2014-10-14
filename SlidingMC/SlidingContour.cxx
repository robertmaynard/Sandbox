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

#include "SlidingContour.h"
#include "DataProcessors.h"
#include "SlidingWorklet.h"

#include <dax/cont/DispatcherMapCell.h>
#include <dax/cont/Timer.h>

//-----------------------------------------------------------------------------
ImageStore::ImageStore( std::string f, int slices)
{

  this->InputData = detail::read_ImageData(f);
  this->DaxGrid = detail::extract_grid_info_from_ImageData(this->InputData);

  const dax::Id3 dims = dax::extentDimensions(this->DaxGrid.GetExtent());
  this->MaxSlices = slices;
  this->ZExtent = dims[2]/slices;
}

//-----------------------------------------------------------------------------
dax::cont::UniformGrid<> ImageStore::dataSlicedAt( int slice ) const
{
  dax::cont::UniformGrid< > slice_of_grid;
  slice_of_grid.SetSpacing( this->DaxGrid.GetSpacing() );
  slice_of_grid.SetOrigin( this->DaxGrid.GetOrigin() );

  dax::Extent3 sub_extent;
  sub_extent = this->DaxGrid.GetExtent();

  sub_extent.Min[2] = this->ZExtent * slice;
  sub_extent.Max[2] = (this->ZExtent * (1 + slice) ) - 1;
  slice_of_grid.SetExtent(sub_extent);

  return slice_of_grid;
}

//-----------------------------------------------------------------------------
dax::cont::ArrayHandle<dax::Scalar> ImageStore::arraySlicedAt( int slice ) const
{
  const dax::Id3 dims = dax::extentDimensions(this->DaxGrid.GetExtent());
  const int offset = dims[0] * dims[1] * (this->ZExtent * slice);
  const int length = dims[0] * dims[1] * this->ZExtent;

  return detail::extract_buffer_from_ImageData(this->InputData, offset, length);
}


//-----------------------------------------------------------------------------
SlidingContour::SlidingContour( ImageStore store, float contourValue):
  ContourValue( contourValue )
{

  typedef dax::cont::DispatcherMapCell<dax::worklet::SlidingContour> DispatcherCount;

  double avg_time = 0;

  const int iterations = store.iterations();
  for(int i=0; i < iterations; ++i)
    {
    dax::cont::Timer<> timer;
    dax::cont::ArrayHandle< dax::Vector3 > triangleHandle;

    //generate a uniform grid that represents a slice of the full grid
    dax::cont::UniformGrid< > slice_of_grid = store.dataSlicedAt(i);

    //generate an ArrayHandle that is offsetted into the full data properly.
    dax::cont::ArrayHandle< dax::Scalar > slice_of_array = store.arraySlicedAt(i);


    //generate a pre-allocated array to store the triangle coordinates.
    //make size per cell is 5 triangles
    std::vector< dax::Vector3 > output_tris_storage( slice_of_grid.GetNumberOfCells() * 5);
    std::fill(output_tris_storage.begin(),
              output_tris_storage.end(),
              dax::Vector3(-1,-1,-1));


    dax::cont::ArrayHandle< dax::Vector3 > output_tris = dax::cont::make_ArrayHandle( output_tris_storage );
    dax::worklet::SlidingContour makeTriangles(contourValue, output_tris);

    DispatcherCount(makeTriangles).Invoke(slice_of_grid, slice_of_array);

    double time = timer.GetElapsedTime();
    avg_time += time;
    }

  std::cout << "avg time to contour: " << avg_time / iterations << std::endl;

}