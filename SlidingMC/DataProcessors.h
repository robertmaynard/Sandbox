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
#ifndef dataProcessors_h
#define dataProcessors_h

#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkImageResample.h>
#include <vtkNrrdReader.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkNew.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>

namespace detail {

static vtkSmartPointer<vtkImageData> read_ImageData(const std::string& file)
{
  std::cout << "reading file: " << file << std::endl;
  vtkNew<vtkNrrdReader> reader;
  reader->SetFileName(file.c_str());
  reader->Update();

  //take ref
  vtkSmartPointer<vtkImageData> image = vtkSmartPointer<vtkImageData>::New();
  vtkImageData *newImageData = vtkImageData::SafeDownCast(reader->GetOutputDataObject(0));
  image.TakeReference( newImageData );
  image->Register(NULL);

  return image;
}

dax::cont::UniformGrid<> extract_grid_info_from_ImageData(
                              vtkSmartPointer<vtkImageData> data)
{
  double origin[3];data->GetOrigin(origin);
  double spacing[3];data->GetSpacing(spacing);
  int extent[6];data->GetExtent(extent);

  //this would be image data
  dax::cont::UniformGrid<> output;
  output.SetOrigin(dax::make_Vector3(origin[0],origin[1],origin[2]));
  output.SetSpacing(dax::make_Vector3(spacing[0],spacing[1],spacing[2]));
  output.SetExtent(dax::make_Id3(extent[0],extent[2],extent[4]),
                   dax::make_Id3(extent[1],extent[3],extent[5]));
  return output;
}

dax::cont::ArrayHandle<dax::Scalar> extract_buffer_from_ImageData(
                              vtkSmartPointer<vtkImageData> data,
                              int offset,
                              int length)
{
 //now set the buffer
 vtkDataArray *newData = data->GetPointData()->GetScalars();
 dax::Scalar* rawBuffer = reinterpret_cast<dax::Scalar*>( newData->GetVoidPointer(0) );
 rawBuffer += offset;
 const dax::Id size = static_cast<dax::Id>(length);
 return dax::cont::make_ArrayHandle(rawBuffer,size);
}

}

#endif