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

#include <vtkTrivialProducer.h>
#include <vtkContourFilter.h>
#include <vtkFloatArray.h>

//-----------------------------------------------------------------------------
ImageStore::ImageStore( std::string f )
{
  this->Image = detail::read_ImageData(f);
}

//-----------------------------------------------------------------------------
ImageProvider::ImageProvider( ImageStore store ):
  Store(store),
  MaxSlices(0),
  ZExtent(0)
{
  this->DaxGrid = detail::extract_grid_info_from_ImageData(this->Store.vtkData());

  const dax::Id3 dims = dax::extentDimensions(this->DaxGrid.GetExtent());

  this->MaxSlices = dims[2] / 2;
  this->ZExtent = 2;
}


//-----------------------------------------------------------------------------
ImageProvider::ImageProvider( ImageStore store, int slices):
  Store(store),
  MaxSlices(slices),
  ZExtent(0)
{
  this->DaxGrid = detail::extract_grid_info_from_ImageData(this->Store.vtkData());

  dax::Extent3 sub_extent;
  sub_extent = this->DaxGrid.GetExtent();

  const dax::Id3 dims = dax::extentDimensions(this->DaxGrid.GetExtent());
  this->ZExtent = dims[2]/this->MaxSlices;
}

//-----------------------------------------------------------------------------
dax::cont::UniformGrid<> ImageProvider::dataSlicedAt( int slice ) const
{
  dax::cont::UniformGrid< > slice_of_grid;
  slice_of_grid.SetSpacing( this->DaxGrid.GetSpacing() );
  slice_of_grid.SetOrigin( this->DaxGrid.GetOrigin() );

  dax::Extent3 sub_extent = this->DaxGrid.GetExtent();
  sub_extent.Min[2] = this->ZExtent * slice;
  sub_extent.Max[2] = (this->ZExtent * (1 + slice) );

  if( sub_extent.Max[2] > this->DaxGrid.GetExtent().Max[2])
    {
    sub_extent.Max[2] = this->DaxGrid.GetExtent().Max[2];
    }

  // std::cout << sub_extent.Min[2] << " to " << sub_extent.Max[2] << std::endl;
  slice_of_grid.SetExtent(sub_extent);
  return slice_of_grid;
}

//-----------------------------------------------------------------------------
dax::cont::ArrayHandle<dax::Scalar> ImageProvider::arraySlicedAt( int slice ) const
{
  const dax::Id3 dims = dax::extentDimensions(this->DaxGrid.GetExtent());
  const int offset = dims[0] * dims[1] * (this->ZExtent * slice);
  const int length = dims[0] * dims[1] * (this->ZExtent+1);

  // std::cout << offset << " to " << offset+length << std::endl;
  return detail::extract_buffer_from_ImageData(this->Store.vtkData(), offset, length);
}

//-----------------------------------------------------------------------------
ClassicContour::ClassicContour( ImageStore store, float contourValue):
  ContourValue( contourValue )
{
  vtkNew<vtkTrivialProducer> producer;
  vtkNew<vtkContourFilter> contour;

  dax::cont::Timer<> alg_timer;
  producer->SetOutput( store.vtkData() );
  contour->SetInputConnection( producer->GetOutputPort() );
  contour->SetNumberOfContours( 1 );
  contour->ComputeNormalsOff();
  contour->ComputeScalarsOff();
  contour->SetValue( 0, contourValue );
  contour->Update();

  //take ref and store
  this->OutputData = vtkSmartPointer<vtkPolyData>::New();
  this->OutputData.TakeReference( contour->GetOutput() );
  this->OutputData->Register(NULL);

  std::cout << "time to contour using vtk: " << alg_timer.GetElapsedTime() << std::endl;
}

//-----------------------------------------------------------------------------
void ClassicContour::write( std::string path ) const
{
  std::cout << "this->vtkData()->GetNumberOfCells() : " <<  this->vtkData()->GetNumberOfCells() << std::endl;
  std::cout << "this->vtkData()->GetNumberOfPoints(): " << this->vtkData()->GetNumberOfPoints() << std::endl;
  detail::write(this->vtkData(), path);
}


//-----------------------------------------------------------------------------
SlidingContour::SlidingContour( ImageProvider provider, float contourValue):
  ContourValue( contourValue )
{

  typedef dax::cont::DispatcherMapCell<dax::worklet::SlidingContour> DispatcherCount;
  typedef dax::cont::DeviceAdapterAlgorithm< DAX_DEFAULT_DEVICE_ADAPTER_TAG > Algorithm;

  typedef dax::cont::UniformGrid<>::PointCoordinatesType::PortalConstExecution EdgeInPortalType;

  dax::cont::Timer<> alg_timer;
  double contour_time = 0;
  double interpolate_time = 0;
  double total_fill_time = 0;

  const int iterations = provider.iterations();
  std::cout << "starting up the sliding contour algorithm with " << iterations << " slices." << std::endl;


  //generate a pre-allocated array to provider the triangle coordinates. this
  //will be reused by each run
  const std::size_t numCells = provider.dataSlicedAt(0).GetNumberOfCells();
  const std::size_t tri_storage_size = numCells * 15;

  vtkSmartPointer<vtkFloatArray> outputTriangles = vtkSmartPointer<vtkFloatArray>::New();
  outputTriangles->SetNumberOfComponents(3);
  outputTriangles->SetNumberOfTuples( 0 );
  std::vector< dax::PointAsEdgeInterpolation > raw_edge_storage( tri_storage_size );

  for(int i=0; i < iterations; ++i)
    {
    dax::cont::Timer<> timer;

    //generate a uniform grid that represents a slice of the full grid
    dax::cont::UniformGrid< > slice_of_grid = provider.dataSlicedAt(i);

    //generate an ArrayHandle that is offsetted into the full data properly.
    dax::cont::ArrayHandle< dax::Scalar > slice_of_array = provider.arraySlicedAt(i);

    // std::cout << "grid num points: " << slice_of_grid.GetNumberOfPoints() << std::endl;
    // std::cout << "array length: " << slice_of_array.GetNumberOfValues() << std::endl;

    dax::worklet::contour_atomic_lock = 0; //clear the lock to zero
    dax::worklet::SlidingContour makeTriangles(contourValue, &raw_edge_storage);

    DispatcherCount(makeTriangles).Invoke(slice_of_grid, slice_of_array);

    const std::size_t numTriangleCoords = dax::worklet::contour_atomic_lock;
    if(numTriangleCoords > 0)
      {
      contour_time += timer.GetElapsedTime(); timer.Reset();
      //now convert the edges to points
      dax::worklet::InterpolateEdgeToPoints<EdgeInPortalType> interpolate( slice_of_grid.GetPointCoordinates().PrepareForInput(),
                                                                           &raw_edge_storage,
                                                                           outputTriangles,
                                                                           numTriangleCoords);
      Algorithm::Schedule(interpolate, numTriangleCoords);
      }

    interpolate_time += timer.GetElapsedTime(); timer.Reset();
    }
  std::cout << "time to contour: " << contour_time << std::endl;
  std::cout << "time to interpolate: " << interpolate_time << std::endl;

  //now we need to move the data to vtkPolyData.
  dax::cont::Timer<> vtk_timer;
  this->OutputData = detail::convert_to_PolyData( outputTriangles.GetPointer() );
  std::cout << "time to convert to vtk: " << vtk_timer.GetElapsedTime() << std::endl;

  std::cout << "time to contour + overhead: " << alg_timer.GetElapsedTime() << std::endl;

}

//-----------------------------------------------------------------------------
void SlidingContour::write( std::string path ) const
{
  std::cout << "this->vtkData()->GetNumberOfCells() : " <<  this->vtkData()->GetNumberOfCells() << std::endl;
  std::cout << "this->vtkData()->GetNumberOfPoints(): " << this->vtkData()->GetNumberOfPoints() << std::endl;
  detail::write(this->vtkData(), path);
}

