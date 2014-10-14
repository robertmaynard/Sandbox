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
ImageStore::ImageStore( std::string f )
{
  this->InputData = detail::read_ImageData(f);
  this->DaxGrid = detail::extract_grid_info_from_ImageData(this->InputData);
  this->DaxArray = detail::extract_buffer_from_ImageData(this->InputData);
}

//-----------------------------------------------------------------------------
SlidingContour::SlidingContour( ImageStore store, float contourValue):
  ContourValue( contourValue )
{

  typedef dax::cont::DispatcherMapCell<dax::worklet::SlidingContour> DispatcherCount;

  double avg_time = 0;
  const int iterations = 16;

  dax::Id3 dims = dax::extentDimensions(store.data().GetExtent());
  const int z_extent = dims[2]/iterations;
  std::cout << "z_extent: " << z_extent << std::endl;

  for(int i=0; i < iterations; ++i)
    {
    dax::cont::Timer<> timer;
    dax::cont::ArrayHandle< dax::Vector3 > triangleHandle;

    //construct the worklet that will be used to do the marching cubes
    dax::worklet::SlidingContour makeTriangles(contourValue, i);

    //generate a uniform grid that represents a slice of the full grid
    dax::cont::UniformGrid< > slice_of_grid;
    slice_of_grid.SetSpacing( store.data().GetSpacing() );
    slice_of_grid.SetOrigin( store.data().GetOrigin() );

    dax::Extent3 sub_extent;
    sub_extent = store.data().GetExtent();

    sub_extent.Min[2] = z_extent * i;
    sub_extent.Max[2] = (z_extent * (1 + i) ) - 1;
    slice_of_grid.SetExtent(sub_extent);

    std::cout << sub_extent.Min[2] << ", " << sub_extent.Max[2] << std::endl;

    dax::Vector3 ploc = slice_of_grid.ComputePointCoordinates( 0 );
    std::cout << "first point position: (" << ploc[0] << ", " << ploc[1] << ", " << ploc[2] << ")" << std::endl;

    // DispatcherCount(makeTriangles).Invoke(store.data(), store.array());

    double time = timer.GetElapsedTime();
    avg_time += time;
    }

  std::cout << "avg time to contour: " << avg_time / iterations << std::endl;

}