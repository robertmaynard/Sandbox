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

#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/DispatcherGenerateInterpolatedCells.h>
#include <dax/cont/DispatcherMapCell.h>
#include <dax/cont/DispatcherMapField.h>
#include <dax/cont/Timer.h>

#include <dax/opengl/TransferToOpenGL.h>

namespace detail
{

  mandle::MandlebulbSurface generateSurface( mandle::MandlebulbVolume& vol,
                            dax::Scalar iteration,
                            dax::cont::ArrayHandle<dax::Id> count)

{
  mandle::MandlebulbSurface surface; //construct surface struct

  // dax::cont::Timer<> timer;

  // //setup the info for the second step
  // dax::worklet::MarchingCubesGenerate generateSurface(iteration);
  // dax::cont::DispatcherGenerateInterpolatedCells<
  //     ::dax::worklet::MarchingCubesGenerate > surfDispacther(count,
  //                                                            generateSurface);

  // surfDispacther.SetRemoveDuplicatePoints(false);

  // //run the second step
  // surfDispacther.Invoke( vol.Grid, surface.Data, vol.EscapeIteration);

  // std::cout << "mc stage 2: " << timer.GetElapsedTime() << " sec" << std::endl;

  // //generate a color for each point based on the escape iteration
  // if(surface.Data.GetNumberOfPoints() > 0)
  //   {
  //   mandle::SurfaceCoords surface_coords(surface.Data);
  //   dax::cont::DispatcherMapField<worklet::ColorsAndNorms> colorNormsDispatcher;
  //   colorNormsDispatcher.Invoke(
  //                 dax::cont::make_ArrayHandleCounting(dax::Id(0),
  //                                         surface.Data.GetNumberOfPoints()),
  //                 surface_coords,
  //                 surface.Norms,
  //                 surface.Colors);
  //   std::cout << "colors & norms: " << timer.GetElapsedTime() << " sec"
  //             << std::endl;
  //   }

  return surface;
}

}


namespace mandle
{

  MandlebulbVolume::MandlebulbVolume( dax::Vector3 origin,
                                      dax::Vector3 spacing,
                                      dax::Extent3 extent )
  {
    this->NumberOfSubGrids = 200;
    dax::Id3 size = dax::extentDimensions(extent);

    const std::size_t z_size_per_sub_grid = size[2] / this->NumberOfSubGrids;
    const std::size_t z_remainder = size[2] % this->NumberOfSubGrids;

    std::cout << "z_size_per_sub_grid:  " <<z_size_per_sub_grid << std::endl;

    for(std::size_t i=0; i < this->NumberOfSubGrids-1; ++i)
      {
      dax::cont::UniformGrid< > subGrid;
      subGrid.SetSpacing( spacing ); //same as the full grid

      //calculate the origin, only z changes
      dax::Vector3 sub_origin = origin;
      sub_origin[2] = origin[2] + ( (i * z_size_per_sub_grid) * spacing[2]);
      subGrid.SetOrigin( sub_origin );

      //calculate out the new extent
      dax::Extent3 sub_extent = extent;
      sub_extent.Min[2] = 0;
      sub_extent.Max[2] = z_size_per_sub_grid -1 ; //account for it being cells
      subGrid.SetExtent(sub_extent);

      this->SubGrids.push_back(subGrid);
      std::cout << "num of cells in grid " << this->SubGrids[i].GetNumberOfCells() << std::endl;
      std::cout << "num of points in grid " << this->SubGrids[i].GetNumberOfPoints() << std::endl;
      }

    //add in the last subgrid
    dax::cont::UniformGrid< > subGrid;
    subGrid.SetSpacing( spacing ); //same as the full grid

    //calculate the origin, only z changes
    dax::Vector3 sub_origin = origin;
    sub_origin[2] = origin[2] + z_remainder +
            ( (this->NumberOfSubGrids-1) * z_size_per_sub_grid * spacing[2]);
    subGrid.SetOrigin( sub_origin );

    //calculate out the new extent
    dax::Extent3 sub_extent = extent;
    sub_extent.Min[2] = 0;
    sub_extent.Max[2] = (z_size_per_sub_grid -1) + z_remainder; //account for it being cells
    subGrid.SetExtent(sub_extent);

    this->SubGrids.push_back(subGrid);

    //now create the rest of the vectors to the same size as the subgrids
    this->PerSliceLowHighs.resize(this->NumberOfSubGrids);
    this->LowHighs.resize(this->NumberOfSubGrids);
    this->EscapeIterations.resize(this->NumberOfSubGrids);
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

      // // //pull everything down to the control side
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


  const std::size_t size = vol.numSubGrids();
  for(std::size_t i = 0; i < size; ++i)
    {
    if(vol.isValidSubGrid(i, iteration))
      {
      dax::cont::ArrayHandle<dax::Id> count;
      dax::cont::DispatcherMapCell< ::worklet::MarchingCubesHLCount >
        classify( ( ::worklet::MarchingCubesHLCount(iteration, vol.subGrid(i),  vol.subEscapes(i) )) );
      classify.Invoke( vol.subGrid(i), vol.subLowHighs(i), count );
      }
    vol.releaseExecMemForSubGrid(i);
    }

  std::cout << "mc stage 1: " << timer.GetElapsedTime() << " sec" << std::endl;

  mandle::MandlebulbSurface surface;
  return surface;
  // return detail::generateSurface(vol,iteration,count);
}

//compute the clip of the volume for a given iteration
mandle::MandlebulbSurface extractCut( mandle::MandlebulbVolume& vol,
                                        dax::Scalar cut_percent,
                                        dax::Scalar iteration )
{

  // dax::Vector3 origin = vol.Grid.GetOrigin();
  // dax::Vector3 spacing = vol.Grid.GetSpacing();
  // dax::Id3 dims = dax::extentCellDimensions(vol.Grid.GetExtent());

  // //slicing at the edges where nothing is causes problems
  // //we are doing z slice so we have to go from positive
  // dax::Vector3 location(origin[0] + spacing[0] * dims[0],
  //                       origin[1] + spacing[1] * dims[1],
  //                       origin[2] + spacing[2] * (dims[2] * cut_percent) );
  // dax::Vector3 normal(0,0,1);

  // //lets extract the clip
  dax::cont::ArrayHandle<dax::Id> count;

  // dax::cont::Timer<> timer;


  // dax::cont::DispatcherMapField< ::worklet::MarchingCubesHLClip >
  //       classify( ( ::worklet::MarchingCubesHLClip(origin, location, normal,
  //                                                  iteration, vol.Grid,
  //                                                  vol.EscapeIteration,
  //                                                  vol.Grid.GetPointCoordinates() )) );

  // classify.Invoke( vol.LowHigh, count );

  // std::cout << "mc stage 1: "  << timer.GetElapsedTime() << std::endl;

  return detail::generateSurface(vol,iteration,count);
}


void bindSurface( mandle::MandlebulbSurface& surface,
                  GLuint& coord,
                  GLuint& color,
                  GLuint& norm )
{
  if(surface.Data.GetNumberOfPoints() == 0)
    return;

  //TransferToOpenGL will do the binding to the given buffers if needed
  dax::opengl::TransferToOpenGL(surface.Data.GetPointCoordinates(), coord);
  dax::opengl::TransferToOpenGL(surface.Colors, color);
  dax::opengl::TransferToOpenGL(surface.Norms, norm);

  //no need to keep the cuda side, as the next re-computation will have
  //redo all the work for all three of these
  surface.Colors.ReleaseResourcesExecution();
  surface.Norms.ReleaseResourcesExecution();

}
