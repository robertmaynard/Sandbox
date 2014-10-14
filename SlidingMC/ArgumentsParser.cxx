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

#include "ArgumentsParser.h"

#include <dax/testing/OptionParser.h>
#include <iostream>
#include <sstream>
#include <string>

enum  optionIndex { UNKNOWN, HELP, FILEPATH, CONTOUR_VALUE};
const dax::testing::option::Descriptor usage[] =
{
  {UNKNOWN,   0,"" , ""    ,      dax::testing::option::Arg::None, "USAGE: example [options]\n\n"
                                                                    "Options:" },
  {HELP,      0,"h" , "help",    dax::testing::option::Arg::None, "  --help, -h  \tPrint usage and exit." },
  {FILEPATH,      0,"", "file",      dax::testing::option::Arg::Optional, "  --file  \t nhdr file to read." },
  {CONTOUR_VALUE,      0,"", "contour",      dax::testing::option::Arg::Optional, "  --contour  \t iso value to contour at." },
  {UNKNOWN,   0,"",  "",         dax::testing::option::Arg::None, "\nExample:\n"
                                                                   " example --file=./test \n"},
  {0,0,0,0,0,0}
};

namespace dax {
namespace benchmark {

//-----------------------------------------------------------------------------
ArgumentsParser::ArgumentsParser():
  File("")
{
}

//-----------------------------------------------------------------------------
ArgumentsParser::~ArgumentsParser()
{
}

//-----------------------------------------------------------------------------
bool ArgumentsParser::parseArguments(int argc, char* argv[])
{

  argc-=(argc>0);
  argv+=(argc>0); // skip program name argv[0] if present

  dax::testing::option::Stats  stats(usage, argc, argv);
  dax::testing::option::Option* options = new dax::testing::option::Option[stats.options_max];
  dax::testing::option::Option* buffer = new dax::testing::option::Option[stats.options_max];
  dax::testing::option::Parser parse(usage, argc, argv, options, buffer);

  if (parse.error())
    {
    delete[] options;
    delete[] buffer;
    return false;
    }

  if (options[HELP] || argc == 0)
    {
    dax::testing::option::printUsage(std::cout, usage);
    delete[] options;
    delete[] buffer;

    return false;
    }

  if ( options[FILEPATH] )
    {
    std::string sarg(options[FILEPATH].last()->arg);
    std::stringstream argstream(sarg);
    argstream >> this->File;
    }

  if ( options[CONTOUR_VALUE] )
    {
    std::string sarg(options[CONTOUR_VALUE].last()->arg);
    std::stringstream argstream(sarg);
    argstream >> this->Contour;
    }

  delete[] options;
  delete[] buffer;
  return true;
}

}
}