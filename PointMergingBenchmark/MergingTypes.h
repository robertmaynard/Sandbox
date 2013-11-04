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

#ifndef __MergingTypes_h
#define __MergingTypes_h

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

#include "Timer.h"
#include "PointGenerator.h"
#include "vtkIncrementalOctreePointLocator.h"
#include "vtkMergePoints.h"
#include "vtkPolyData.h"
#include "vtkNew.h"



namespace merging
{
typedef PointGenerator::const_iterator PointIterator;
typedef std::vector< vtkIdType > TopologyType;

  TopologyType make_topo(PointGenerator& gen)
  {
    TopologyType topo;
    topo.reserve(gen.size());
    return topo;
  }

//use vtk point locator
void Locator(PointGenerator& generator)
{
  vtkNew<vtkPoints> ps;
  ps->SetDataTypeToFloat();
  vtkNew<vtkIncrementalOctreePointLocator> locator;

  double bounds[6] = {0,10000000,0,10000000,0,10000000,};
  locator->InitPointInsertion(ps.GetPointer(),bounds);

  TopologyType topo = make_topo(generator);

  vtkIdType id=0;
  for(PointIterator it = generator.begin();
      it != generator.end();
      ++it)
    {
    const double xyz[3]= {(*it).X, (*it).Y, (*it).Z};

    locator->InsertUniquePoint(xyz, id);
    topo.push_back(id);
    }

  std::cout << "num points: " << ps->GetNumberOfPoints() << std::endl;
}

//use vtk point locator
void LocatorMP(PointGenerator& generator)
{
  vtkNew<vtkPoints> ps;
  ps->SetDataTypeToFloat();
  vtkNew<vtkMergePoints> locator;

  double bounds[6] = {0,10000000,0,10000000,0,10000000,};
  locator->InitPointInsertion(ps.GetPointer(),bounds);

  TopologyType topo = make_topo(generator);

  vtkIdType id=0;
  for(PointIterator it = generator.begin();
      it != generator.end();
      ++it)
    {
    const double xyz[3]= {(*it).X, (*it).Y, (*it).Z};

    locator->InsertUniquePoint(xyz, id);
    topo.push_back(id);
    }

  std::cout << "num points: " << ps->GetNumberOfPoints() << std::endl;
}


//This was SLOW, not point benchmarking it
//use a vector and than convert into set to find subset
void VectorIntoSet(PointGenerator& generator)
{
  std::vector< generator::Point > storage;
  storage.reserve( generator.size() );

  TopologyType topo = make_topo(generator);

  for(PointIterator it = generator.begin();
      it != generator.end();
      ++it)
    {
    //insert point and push_back the position of the point
    storage.push_back( *it );
    }

  //find the reduced set
  std::set< generator::Point > reduced_storage(storage.begin(), storage.end());

  //fill topology with the reduced sets position
  TopologyType::iterator i = topo.begin();
  std::set< generator::Point >::const_iterator first_point = reduced_storage.begin();
  for(std::vector< generator::Point >::const_iterator it = storage.begin();
      it != storage.end();
      ++it, ++i)
    {
    *i = std::distance(first_point, reduced_storage.lower_bound(*it));
    }


  //construct memory vtk can use for points
  std::vector< generator::Point > final_points;
  final_points.reserve(reduced_storage.size());
  std::copy(reduced_storage.begin(), reduced_storage.end(), final_points.begin());
}

void VectorLowerBounds(PointGenerator& generator)
{
  std::vector< generator::Point > storage;
  storage.reserve( generator.size() );

  TopologyType topo = make_topo(generator);
  for(PointIterator it = generator.begin();
      it != generator.end();
      ++it)
    {
    storage.push_back(*it );
    }

  std::vector< generator::Point > reduced_storage(storage.begin(), storage.end());
  std::sort( reduced_storage.begin(), reduced_storage.end() );

  std::vector< generator::Point >::iterator new_end = std::unique( reduced_storage.begin(), reduced_storage.end() );
  reduced_storage.resize( std::distance( reduced_storage.begin(), new_end) ); //remove all unused points

  //fill topology with the reduced sets position
  TopologyType::iterator i = topo.begin();
  std::vector< generator::Point >::const_iterator first_point = reduced_storage.begin();
  std::vector< generator::Point >::const_iterator lb_result;
  for(std::vector< generator::Point >::const_iterator it = storage.begin();
      it != storage.end();
      ++it, ++i)
    {
    lb_result = std::lower_bound(reduced_storage.begin(),reduced_storage.end(),*it);
    *i = std::distance(first_point,lb_result);
    }

 std::cout << "num points: " << reduced_storage.size() << std::endl;
}



}

#endif
