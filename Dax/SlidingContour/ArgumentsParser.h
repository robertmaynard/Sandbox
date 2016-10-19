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
#ifndef slidingmc__argumentsParser_h
#define slidingmc__argumentsParser_h

#include <string>

namespace dax {
namespace benchmark {

class ArgumentsParser
{
public:
  ArgumentsParser();
  virtual ~ArgumentsParser();

  bool parseArguments(int argc, char* argv[]);

  std::string file() const
    { return this->File; }

  float contourValue() const
    { return this->Contour; }

  float sliceCount() const
    { return this->SliceCount; }

private:
  std::string File;
  float Contour;
  int SliceCount;
};

}}
#endif