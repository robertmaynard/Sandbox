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
#include <map>
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
typedef std::vector< vtkIdType > TopologyType;

template<typename PointGeneratorType>
TopologyType make_topo(PointGeneratorType& gen)
{
  TopologyType topo;
  topo.reserve(gen.size());
  return topo;
}

//use vtk point locator
template<typename PointGeneratorType>
void Locator(PointGeneratorType& generator)
{
  typedef typename PointGeneratorType::const_iterator PointIterator;
  typedef typename PointGeneratorType::PointType PointType;

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
template<typename PointGeneratorType>
void LocatorMP(PointGeneratorType& generator)
{
  typedef typename PointGeneratorType::const_iterator PointIterator;
  typedef typename PointGeneratorType::PointType PointType;

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

//iterate over the points inserting a key value into a map
//where each key is the point and the value is index
template<typename PointGeneratorType>
void VectorIntoDict(PointGeneratorType& generator)
{
  typedef typename PointGeneratorType::const_iterator PointIterator;
  typedef typename PointGeneratorType::PointType PointType;

  typedef std::pair< PointType, int > value_type;
  typedef std::map< PointType, int > dict_type;
  typedef typename std::map< PointType, int >::iterator iterator;
  typedef typename std::map< PointType, int >::const_iterator const_iterator;
  typedef std::pair< iterator, bool > insert_ret_type;

  //construct the dict
  dict_type idDict;

  //construct the topology array
  TopologyType topo = make_topo(generator);

  {
  TopologyType::iterator i = topo.begin();
  int index=0;
  for(PointIterator it = generator.begin();
      it != generator.end();
      ++it, ++i)
    {
    //insert point and push_back the position of the point
    insert_ret_type ret = idDict.insert( value_type(*it,index) );

    //set the topology for this point to be the correct index value
    const int properIndexValue = ret.first->second;
    *i = properIndexValue;

    if(ret.second)
      {//only increment index if we are inserting a key that didn't already exist
      ++index;
      }

    }
  } //release all temp memory for constructing the dict and topology


  //now construct the reduced point set in a format that vtk can use for points
  std::vector< PointType > reduced_storage;
  reduced_storage.reserve(idDict.size());

  for(const_iterator i=idDict.begin(); i != idDict.end(); ++i)
    {
    reduced_storage.push_back( i->first );
    }

  std::cout << "num points: " << reduced_storage.size() << " using map "<< std::endl;

}

template<typename PointGeneratorType>
void VectorLowerBounds(PointGeneratorType& generator)
{
  typedef typename PointGeneratorType::const_iterator PointIterator;
  typedef typename PointGeneratorType::PointType PointType;

  std::vector< PointType > storage;
  storage.reserve( generator.size() );

  TopologyType topo = make_topo(generator);
  for(PointIterator it = generator.begin();
      it != generator.end();
      ++it)
    {
    storage.push_back(*it );
    }

  std::vector< PointType > reduced_storage(storage.begin(), storage.end());
  std::sort( reduced_storage.begin(), reduced_storage.end() );

  typedef typename std::vector< PointType >::iterator iterator;
  typedef typename std::vector< PointType >::const_iterator const_iterator;

  iterator new_end = std::unique( reduced_storage.begin(), reduced_storage.end() );
  reduced_storage.resize( std::distance( reduced_storage.begin(), new_end) ); //remove all unused points

  //fill topology with the reduced sets position
  TopologyType::iterator i = topo.begin();
  const_iterator first_point = reduced_storage.begin();
  const_iterator lb_result;
  for(const_iterator it = storage.begin(); it != storage.end(); ++it, ++i)
    {
    lb_result = std::lower_bound(reduced_storage.begin(),reduced_storage.end(),*it);
    *i = std::distance(first_point,lb_result);
    }

 std::cout << "num points: " << reduced_storage.size() << " using lower bounds" << std::endl;
}

}

#endif
