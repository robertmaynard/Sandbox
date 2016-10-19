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

#include "Mandlebulb.h"
#include "Worklets.h"

#include <iostream>
#include <limits>

#include <dax/cont/UniformGrid.h>

#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/DispatcherGenerateInterpolatedCells.h>
#include <dax/cont/DispatcherMapCell.h>
#include <dax/cont/DispatcherMapField.h>
#include <dax/cont/Timer.h>

#include <dax/opengl/TransferToOpenGL.h>

dax::Id generateSurface( const dax::cont::UniformGrid< >& vol,
                         const dax::cont::ArrayHandle<dax::Scalar>& escape,
                         dax::Scalar iteration,
                         dax::cont::ArrayHandle<dax::Id> count,
                         mandle::MandlebulbSurface& surface)

{
  dax::cont::UnstructuredGrid< dax::CellTagTriangle > tempGrid;

  //setup the info for the second step
  ::worklet::MarchingCubesHLGenerate genSurface(iteration);
  ::dax::cont::DispatcherGenerateInterpolatedCells<
     ::worklet::MarchingCubesHLGenerate > surfDispatch(count,genSurface);

  surfDispatch.SetRemoveDuplicatePoints(false);

  surfDispatch.Invoke( vol, tempGrid, escape);

  surface.Grids.push_back(tempGrid);
  return tempGrid.GetNumberOfCells();
}


namespace mandle
{

  MandlebulbVolume::MandlebulbVolume( dax::Vector3 origin,
                                      dax::Vector3 spacing,
                                      dax::Extent3 extent )
{
  this->Origin = origin;
  this->Spacing = spacing;
  this->Extent = extent;

  //Todo make the subGridPerDim a far smarter algorithm based
  //on the extents being passed in.
  const std::size_t subGridPerDim = 16;

  dax::Id3 size = dax::extentDimensions(extent);

  dax::Id3 size_per_sub_grid, remainder;
  for(std::size_t dim=0; dim < 3; ++dim)
    {
    size_per_sub_grid[dim] = size[dim] / subGridPerDim;
    remainder[dim] = size[dim] % subGridPerDim;
    }

  dax::Id3 my_remainder;

  for(std::size_t k=0; k < subGridPerDim; ++k)
    {
    if(k == subGridPerDim-1)
      { my_remainder[2]=remainder[2]; }
    else { my_remainder[2]=0; }

    for(std::size_t j=0; j < subGridPerDim; ++j)
      {
      if(j == subGridPerDim-1)
        { my_remainder[1]=remainder[1]; }
      else { my_remainder[1]=0; }

      for(std::size_t i=0; i < subGridPerDim; ++i)
        {
        if(i == subGridPerDim-1)
          { my_remainder[0]=remainder[0]; }
        else { my_remainder[0]=0; }

        dax::cont::UniformGrid< > subGrid;
        subGrid.SetSpacing( spacing ); //same as the full grid

        //calculate the origin
        dax::Vector3 offset(i * size_per_sub_grid[0],
                            j * size_per_sub_grid[1],
                            k * size_per_sub_grid[2]);
        dax::Vector3 sub_origin = origin + ( offset * spacing);

        sub_origin[0] +=  my_remainder[0];
        sub_origin[1] +=  my_remainder[1];
        sub_origin[2] +=  my_remainder[2];
        subGrid.SetOrigin( sub_origin );

        //calculate out the new extent
        dax::Extent3 sub_extent;
        sub_extent.Min = dax::Id3(0,0,0);
        sub_extent.Max = size_per_sub_grid - dax::make_Id3(1,1,1) + my_remainder; //account for it being cells
        subGrid.SetExtent(sub_extent);

        this->SubGrids.push_back(subGrid);
        }
      }
    }
  std::cout << "Number of subgrids " << this->SubGrids.size() << std::endl;

  //now create the rest of the vectors to the same size as the subgrids
  this->PerSliceLowHighs.resize( this->SubGrids.size() );
  this->LowHighs.resize( this->SubGrids.size() );
  this->EscapeIterations.resize( this->SubGrids.size() );
}

void MandlebulbVolume::compute()
{
  dax::cont::Timer<> timer;

  dax::Id numPoints = 0;

  typedef std::vector< dax::cont::UniformGrid< > >::const_iterator gridIt;
  typedef std::vector< dax::cont::ArrayHandle<dax::Scalar> >::iterator escIt;
  typedef std::vector< dax::cont::ArrayHandle<dax::Vector2> >::iterator lhIt;

  escIt escape = this->EscapeIterations.begin();
  lhIt lowhigh = this->LowHighs.begin();

  for(gridIt grid = this->SubGrids.begin();
      grid != this->SubGrids.end();
      ++grid, ++escape, ++lowhigh)
    {
    dax::cont::UniformGrid< >  g = *grid;
    numPoints += g.GetNumberOfPoints();

    //compute the escape iterations for each point in the grid
    dax::cont::ArrayHandle<dax::Scalar> e;
    dax::cont::DispatcherMapField< worklet::Mandlebulb >().Invoke(
                              g.GetPointCoordinates(), e );
    (*escape) = e;

    //compute the low highs for each sub grid
    dax::cont::ArrayHandle<dax::Vector2> lh;
    dax::cont::DispatcherMapCell< ::worklet::FindLowHigh >().Invoke(
                                                g, e, lh);
    (*lowhigh)=lh;

    //pull everything down to the control side
    (*escape).GetPortalConstControl();
    (*lowhigh).GetPortalConstControl();

    //release the exec resources to free memory
    (*escape).ReleaseResourcesExecution();
    (*lowhigh).ReleaseResourcesExecution();
    }

  std::cout << "Compute Mandelbulb Field: " << timer.GetElapsedTime()
  << " sec (" << numPoints << " points)"
  << std::endl;


  //compute the low high per sub grid, skip empty subgrids
  std::vector< dax::Vector2 >::iterator perSlice = this->PerSliceLowHighs.begin();
  gridIt grid = this->SubGrids.begin();

  for(lowhigh = this->LowHighs.begin(); lowhigh != this->LowHighs.end(); ++lowhigh, ++perSlice, ++grid)
    {
    if((*grid).GetNumberOfCells() > 0)
      {
      const dax::Id size = (*lowhigh).GetNumberOfValues();
      dax::Vector2 lh;
      lh[0] = (*lowhigh).GetPortalConstControl().Get(0)[0];
      lh[1] = (*lowhigh).GetPortalConstControl().Get(0)[1];
      for(dax::Id i=1; i < size; ++i)
        {
        dax::Vector2 v = (*lowhigh).GetPortalConstControl().Get(i);
        lh[0] = std::min(v[0],lh[0]);
        lh[1] = std::max(v[1],lh[1]);
        }
      (*perSlice) = lh;
      }
    else
      {
      //invalid cell so make the low high impossible to match
      (*perSlice) = dax::Vector2(1,-1);
      }
    }
}

bool MandlebulbVolume::isValidSubGrid(std::size_t index, dax::Scalar value)
  {
  return this->PerSliceLowHighs[index][0] <= value &&
         this->PerSliceLowHighs[index][1] >= value;
  }

bool MandlebulbVolume::releaseExecMemForSubGrid(std::size_t index)
  {
  this->LowHighs[index].ReleaseResourcesExecution();
  this->EscapeIterations[index].ReleaseResourcesExecution();
  return true;
  }
}



//compute the mandlebulbs values for each point
mandle::MandlebulbVolume computeMandlebulb( dax::Vector3 origin,
                                          dax::Vector3 spacing,
                                          dax::Extent3 extent)
{
  dax::cont::Timer<> timer;

  //construct the dataset with the given origin, spacing, and extent
  mandle::MandlebulbVolume vol(origin,spacing,extent);
  vol.compute();
  return vol;
}

//compute the surface of the mandlebulb for a given iteration
mandle::MandlebulbSurface extractSurface( mandle::MandlebulbVolume& vol,
                                          dax::Scalar iteration )
{
  //find the default device adapter
  typedef DAX_DEFAULT_DEVICE_ADAPTER_TAG AdapterTag;

  //Make it easy to call the DeviceAdapter with the right tag
  typedef dax::cont::DeviceAdapterAlgorithm<AdapterTag> DeviceAdapter;

  dax::cont::Timer<> timer;

  typedef std::vector< dax::cont::UniformGrid< > >::const_iterator gridIt;
  typedef std::vector< dax::cont::ArrayHandle<dax::Scalar> >::const_iterator escIt;
  typedef std::vector< dax::cont::ArrayHandle<dax::Vector2> >::const_iterator lhIt;

  std::size_t numValidSubGrids=0, totalSubGrids=0;
  std::size_t totalValidCells=0, totalCells=0;
  std::size_t totalTris=0;
  double elapsedTime=0, elapsedTime2=0;

  const std::size_t size = vol.numSubGrids();

  //first pass we release any memory from the gpu that we can
  //this will make sure we don't overallocate. In the future we need a better
  //caching scheme than this.
  for(std::size_t i = 0; i < size; ++i, ++totalSubGrids)
    {
    if(!vol.isValidSubGrid(i, iteration))
      { vol.releaseExecMemForSubGrid(i); }
    }

  //now extract the surface on the valid sub grids
  mandle::MandlebulbSurface surface;
  for(std::size_t i = 0; i < size; ++i, ++totalSubGrids)
    {
    if(vol.isValidSubGrid(i, iteration))
      {
      timer.Reset();
      ++numValidSubGrids;
      dax::cont::ArrayHandle<dax::Id> count;
      dax::cont::DispatcherMapCell< ::worklet::MarchingCubesHLCount >
        classify( ( ::worklet::MarchingCubesHLCount(iteration, vol.subGrid(i),  vol.subEscapes(i) )) );
      classify.Invoke( vol.subGrid(i), vol.subLowHighs(i), count );
      elapsedTime += timer.GetElapsedTime();

      //debug, has to be done before generate surface
        {
        dax::cont::ArrayHandle<dax::Id> scannedNewCellCounts;
        DeviceAdapter::StreamCompact(count, scannedNewCellCounts);
        totalValidCells += scannedNewCellCounts.GetNumberOfValues();
        }

      //now time to generate the surface
      timer.Reset();
      totalTris += generateSurface(vol.subGrid(i), vol.subEscapes(i), iteration, count, surface);
      elapsedTime2 += timer.GetElapsedTime();
      }
    totalCells += vol.subGrid(i).GetNumberOfCells();
    }

  std::cout << "mc stage 1: " << elapsedTime  << " sec" << std::endl;
  std::cout << (numValidSubGrids/(float)totalSubGrids * 100) << "% of the subgrids are valid " << std::endl;
  std::cout << (totalValidCells/(float)totalCells * 100) << "% of the cells are valid " << std::endl;
  std::cout << totalValidCells<< " valid cells" << std::endl;

  std::cout << "mc stage 2: " << elapsedTime2  << " sec" << std::endl;
  std::cout << "Total number of triangles in the surface: " << totalTris << std::endl;
  return surface;
}

//compute the clip of the volume for a given iteration
mandle::MandlebulbSurface extractCut( mandle::MandlebulbVolume& vol,
                                      dax::Scalar cut_percent,
                                      dax::Scalar iteration )
{

  dax::Vector3 origin = vol.getOrigin();
  dax::Vector3 spacing = vol.getSpacing();
  dax::Id3 dims = dax::extentCellDimensions(vol.getExtent());

  //slicing at the edges where nothing is causes problems
  //we are doing z slice so we have to go from positive
  dax::Vector3 location(origin[0] + spacing[0] * dims[0],
                        origin[1] + spacing[1] * dims[1],
                        origin[2] + spacing[2] * (dims[2] * cut_percent) );
  dax::Vector3 normal(0,0,1);

  //find the default device adapter
  typedef DAX_DEFAULT_DEVICE_ADAPTER_TAG AdapterTag;

  //Make it easy to call the DeviceAdapter with the right tag
  typedef dax::cont::DeviceAdapterAlgorithm<AdapterTag> DeviceAdapter;

  dax::cont::Timer<> timer;

  typedef std::vector< dax::cont::UniformGrid< > >::const_iterator gridIt;
  typedef std::vector< dax::cont::ArrayHandle<dax::Scalar> >::const_iterator escIt;
  typedef std::vector< dax::cont::ArrayHandle<dax::Vector2> >::const_iterator lhIt;

  std::size_t numValidSubGrids=0, totalSubGrids=0;
  std::size_t totalValidCells=0, totalCells=0;
  double elapsedTime=0;

  const std::size_t size = vol.numSubGrids();

  //first pass we release any memory from the gpu that we can
  //this will make sure we don't overallocate. In the future we need a better
  //caching scheme than this.
  for(std::size_t i = 0; i < size; ++i, ++totalSubGrids)
    {
    if(!vol.isValidSubGrid(i, iteration))
      { vol.releaseExecMemForSubGrid(i); }
    }

  //now extract the surface on the valid sub grids
  mandle::MandlebulbSurface surface;
  for(std::size_t i = 0; i < size; ++i, ++totalSubGrids)
    {
    if(vol.isValidSubGrid(i, iteration))
      {
      timer.Reset();
      ++numValidSubGrids;
      dax::cont::ArrayHandle<dax::Id> count;

      dax::cont::DispatcherMapField< ::worklet::MarchingCubesHLClip >
            classify( ( ::worklet::MarchingCubesHLClip(origin, location, normal,
                                                       iteration, vol.subGrid(i),
                                                       vol.subEscapes(i),
                                                       vol.subGrid(i).GetPointCoordinates() )) );

      classify.Invoke( vol.subLowHighs(i), count );
      elapsedTime += timer.GetElapsedTime();

      //debug
        {
        dax::cont::ArrayHandle<dax::Id> scannedNewCellCounts;
        DeviceAdapter::StreamCompact(count, scannedNewCellCounts);
        totalValidCells += scannedNewCellCounts.GetNumberOfValues();
        }

      //now time to generate the surface
      generateSurface(vol.subGrid(i), vol.subEscapes(i), iteration, count, surface);
      }
    totalCells += vol.subGrid(i).GetNumberOfCells();
    }

  std::cout << "mc stage 1: " << elapsedTime  << " sec" << std::endl;
  std::cout << (numValidSubGrids/(float)totalSubGrids * 100) << "% of the subgrids are valid " << std::endl;
  std::cout << (totalValidCells/(float)totalCells * 100) << "% of the cells are valid " << std::endl;

  return surface;
}


void bindSurface( mandle::MandlebulbSurface& surface,
                  GLuint& coord,
                  GLuint& color,
                  GLuint& norm )
{
  if(surface.Points.GetNumberOfValues() == 0)
    return;

  //TransferToOpenGL will do the binding to the given buffers if needed
  dax::opengl::TransferToOpenGL(surface.Points, coord);
  dax::opengl::TransferToOpenGL(surface.Colors, color);
  dax::opengl::TransferToOpenGL(surface.Norms, norm);

  //no need to keep the cuda side, as the next re-computation will have
  //redo all the work for all three of these
  surface.Colors.ReleaseResourcesExecution();
  surface.Norms.ReleaseResourcesExecution();

}
