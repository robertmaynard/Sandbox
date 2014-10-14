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
#ifndef slidingcontour_h
#define slidingcontour_h

#include <string>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <vtkSmartPointer.h>

class vtkImageData;

class ImageStore
{
public:
  ImageStore( std::string f);

  dax::cont::UniformGrid<> data() const { return this->DaxGrid; }
  dax::cont::ArrayHandle<dax::Scalar> array() const { return this->DaxArray; }

private:
  vtkSmartPointer< vtkImageData > InputData;
  dax::cont::UniformGrid<> DaxGrid;
  dax::cont::ArrayHandle<dax::Scalar> DaxArray;

};

class SlidingContour
{
public:
  SlidingContour( ImageStore store, float contourValue);


private:
  float ContourValue;
};


#endif