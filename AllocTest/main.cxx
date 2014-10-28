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
#include <algorithm>

#include "Timer.h"
#define NUMBER_OF_FRAMES 4096
#define SINGLE_FRAME_SIZE (1024.0*1024.0)
#define FULL_BLOCK_SIZE (4096.0*1024.0*1024.0)

double single_allocation()
{
  Timer time;
  int *x = new int[ static_cast<long long>(FULL_BLOCK_SIZE) ];
  const double t = time.GetElapsedTime();

  //used to make sure the compiler doesn't optimize the allocate out
  time.Reset();
  for(long long i=0; i < FULL_BLOCK_SIZE; ++i)
    {
    x[i] = i;
    }
  std::cout << "set time: " << time.GetElapsedTime() << std::endl;
  delete[] x;

  return t;
}

double multiple_allocations()
{
  Timer time;
  int **table= new int*[static_cast<long long>(NUMBER_OF_FRAMES)];
  for(int i=0; i < NUMBER_OF_FRAMES; ++i)
    {
    table[i] = new int[ static_cast<long long>(SINGLE_FRAME_SIZE) ];
    }
  const double t = time.GetElapsedTime();

  //used to make sure the compiler doesn't optimize the allocate out
  time.Reset();
  for(int i=0; i < NUMBER_OF_FRAMES; ++i)
    {
    for(long long j=0; j <  static_cast<long long>(SINGLE_FRAME_SIZE); ++j)
      {
      table[i][j] = j;
      }
    }
  std::cout << "set time: " << time.GetElapsedTime() << std::endl;

  for(int i=0; i < NUMBER_OF_FRAMES; ++i)
    {
    delete[] table[i];
    }
  delete[] table;

  return t;

}

double inverted_multiple_allocations()
{
  Timer time;
  int **table= new int*[static_cast<long long>(SINGLE_FRAME_SIZE)];
  for(int i=0; i < SINGLE_FRAME_SIZE; ++i)
    {
    table[i] = new int[ static_cast<long long>(NUMBER_OF_FRAMES) ];
    }
  const double t = time.GetElapsedTime();

  //used to make sure the compiler doesn't optimize the allocate out
  for(int i=0; i < SINGLE_FRAME_SIZE; ++i)
    {
    std::memset(table[i], 'a', NUMBER_OF_FRAMES);
    }

  for(int i=0; i < SINGLE_FRAME_SIZE; ++i)
    {
    delete[] table[i];
    }
  delete[] table;

  return t;

}

int main()
{


  double single_time = single_allocation();
  double multiple_time = multiple_allocations();
  // double inv_multiple_time = inverted_multiple_allocations();

  double size = FULL_BLOCK_SIZE / (1024.0*1024.0*1024.0);
  std::cout << "Benchmarking time to allocate: " << size  << "GB" << std::endl;
  std::cout << "Time for a single allocation is: " << single_time << std::endl;
  std::cout << "Time for a multiple allocation is: " << multiple_time << std::endl;
  // std::cout << "Time for a inverted multiple allocation is: " << inv_multiple_time << std::endl;
  return 0;
}