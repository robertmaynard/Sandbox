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

#include <vtkCellArray.h>
#include <vtkDataArray.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkImageResample.h>
#include <vtkNew.h>
#include <vtkNrrdReader.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkTrivialProducer.h>
#include <vtkXMLPolyDataWriter.h>

#include <dax/cont/Timer.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>

namespace detail {

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> convert_to_PolyData(vtkFloatArray* triangle_points)
{
  vtkSmartPointer<vtkPolyData> data = vtkSmartPointer<vtkPolyData>::New();
  vtkNew< vtkPoints > points;
  vtkNew< vtkCellArray > cells;

  // setup the points
  points->SetData(triangle_points);

  //write the cell array
  dax::cont::Timer<> timer;
  const std::size_t num_cells = triangle_points->GetNumberOfTuples()/3;
  cells->Allocate(num_cells * 4);
  std::cout << "cell allocation time: " << timer.GetElapsedTime() << std::endl;
  timer.Reset();

  vtkIdType* cellPointer = cells->GetPointer();
  vtkIdType index = 0;
  //break out the unrolled loop
  for(vtkIdType i=0; i < num_cells; ++i, index +=3)
    {
    *cellPointer = 3; ++cellPointer;
    *cellPointer = index; ++cellPointer;
    *cellPointer = index+1; ++cellPointer;
    *cellPointer = index+2; ++cellPointer;
    }

  std::cout << "cell insertion time: " << timer.GetElapsedTime() << std::endl;

  //set up the polyData
  data->SetPoints( points.GetPointer() );
  data->SetPolys( cells.GetPointer() );
  return data;
}

//-----------------------------------------------------------------------------
void write(vtkSmartPointer<vtkPolyData> data, std::string path)
{
  vtkNew<vtkTrivialProducer> producer;
  vtkNew<vtkXMLPolyDataWriter> writer;

  producer->SetOutput( data );
  writer->SetInputConnection( producer->GetOutputPort() );
  writer->SetFileName( path.c_str() );
  // writer->SetDataModeToAscii();
  writer->Write();
}


}

#endif