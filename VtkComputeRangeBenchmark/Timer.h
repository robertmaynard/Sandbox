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

#ifndef __MyTimer_h
#define __MyTimer_h

///Taken from DAX serial timer
#ifndef _WIN32
#include <limits.h>
#include <sys/time.h>
#include <unistd.h>
#endif

class Timer
{
public:
  /// When a timer is constructed, all threads are synchronized and the
  /// current time is marked so that GetElapsedTime returns the number of
  /// seconds elapsed since the construction.
  Timer()
  {
    this->Reset();
  }

  /// Resets the timer. All further calls to GetElapsedTime will report the
  /// number of seconds elapsed since the call to this. This method
  /// synchronizes all asynchronous operations.
  ///
  void Reset()
  {
    this->StartTime = this->GetCurrentTime();
  }

  /// Returns the elapsed time in seconds between the construction of this
  /// class or the last call to Reset and the time this function is called. The
  /// time returned is measured in wall time. GetElapsedTime may be called any
  /// number of times to get the progressive time. This method synchronizes all
  /// asynchronous operations.
  ///
  float GetElapsedTime()
  {
    TimeStamp currentTime = this->GetCurrentTime();

    float elapsedTime;
    elapsedTime = currentTime.Seconds - this->StartTime.Seconds;
    elapsedTime += ((currentTime.Microseconds - this->StartTime.Microseconds)
                    /float(1000000));

    return elapsedTime;
  }
  struct TimeStamp {
    long long Seconds;
    long long Microseconds;
  };
  TimeStamp StartTime;

  TimeStamp GetCurrentTime()
  {
    TimeStamp retval;
#ifdef _WIN32
    timeb currentTime;
    ::ftime(&currentTime);
    retval.Seconds = currentTime.time;
    retval.Microseconds = 1000*currentTime.millitm;
#else
    timeval currentTime;
    gettimeofday(&currentTime, NULL);
    retval.Seconds = currentTime.tv_sec;
    retval.Microseconds = currentTime.tv_usec;
#endif
    return retval;
  }
};

#endif
