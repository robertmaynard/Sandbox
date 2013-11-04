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

#ifndef __PointGenerator_h
#define __PointGenerator_h

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <sys/time.h>

namespace generator
{

  //light weight float struct that can do comparisons
  //can be overloaded to do point no exact point merging
  //if we wanted
  struct Point
  {
    Point():
    X(0),
    Y(0),
    Z(0)
    {
    }

    Point(float x, float y, float z):
    X(x),
    Y(y),
    Z(z)
    {
    }

    Point( const Point& other):
    X(other.X),
    Y(other.Y),
    Z(other.Z)
    {
    }

    bool operator<(const Point& other) const
    {
      return (X < other.X ) ||
             (X == other.X && Y < other.Y) ||
             (X == other.X && Y == other.Y && Z < other.Z);
    }

    bool operator==(const Point& other) const
    {
      return (X == other.X && Y == other.Y && Z  == other.Z);
    }

    Point& operator=(const Point& other)
    {
      X = other.X; Y = other.Y; Z = other.Z;
      return *this;
    }

    float X,Y,Z;
  };

}

class PointGenerator
{
public:
  typedef std::vector< generator::Point  >::const_iterator const_iterator;

  PointGenerator(float ratio, int num_points)
  {
  float unique_points = num_points * (1.0 - ratio);

  //we need something fancier for this
  Points.reserve(num_points);
  srand (time(NULL));
  for(int i=0; i < unique_points; ++i)
    {
    float x = (int)rand() % (int)unique_points;
    float y = (int)rand() % (int)unique_points;
    float z = (int)rand() % (int)unique_points;
    Points.push_back( generator::Point(x,y,z) );
    }

  for(int i=  unique_points; i < num_points; ++i)
  {
    timeval currentTime;
    gettimeofday(&currentTime, NULL);
    int index =  (int)currentTime.tv_sec % (int)unique_points;
    Points.push_back( generator::Point(index,index,index) );
  }

  //no lets shuffle them all
  std::random_shuffle(Points.begin(),Points.end());

  }

  int size() const
  {
  return Points.size();
  }

  const_iterator begin() const { return Points.begin(); }
  const_iterator end() const { return Points.end(); }



private:
  std::vector< generator::Point  > Points;
};

#endif
