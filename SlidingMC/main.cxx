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

#include <iostream>

#include "ArgumentsParser.h"
#include "SlidingContour.h"

int main(int argc, char* argv[])
  {
  dax::benchmark::ArgumentsParser parser;
  if (!parser.parseArguments(argc, argv))
    {
    return 1;
    }

  std::string file = parser.file();
  float contourValue = parser.contourValue();
  int sliceCount = parser.sliceCount();

  ImageStore s(file);

  // use vtk to get a benchmark size
  {
  ClassicContour cc(s, contourValue);
  // cc.write("vtkoutput.vtp"); //write the result to disk
  } //release the vtk algorithm & memory

  {
  ImageProvider p = ImageProvider(s, sliceCount);
  SlidingContour contour = SlidingContour(p, contourValue);
  contour.write("daxoutput.vtp"); //write the result to disk
  }

  return 0;
}