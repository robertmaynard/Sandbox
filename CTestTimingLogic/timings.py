#!/usr/bin/env python

import math

threshold = 1
mult = 1 # 0.25 might be more sane
track_after_times_failed = 7

def compute(currentTime, mean, sd):
  if(sd < threshold):
    sd = threshold
  weight = 0.3
  newMean = (1.0-weight)*mean + weight*currentTime
  newSD = math.sqrt((1.0-weight)*sd*sd + weight*(currentTime-newMean)* (currentTime-newMean))
  return newMean, newSD

def is_failing(time, mean, sd):
  if(sd < threshold):
    sd = threshold
  failure = (mean + mult * sd)
  print "time[{0}] > (mean[{1}] + mult[{2}] * sd[{3}]) [{4}]".format(time,mean,mult,sd,failure)
  return time > failure



def run(name, timings, mean, sd):
  print name
  times_failed = 0
  for t in timings:
    failed = is_failing(t, mean, sd)
    print t, " is failing => ", failed
    if failed:
      times_failed = times_failed + 1
    else:
      times_failed = 0

    if times_failed > 0 and times_failed <= track_after_times_failed:
      pass
    else:
      mean, sd = compute(t, mean, sd)
      print "new mean,sd => ", (mean,sd)
  print "final mean,sd => ", (mean,sd)

# Test: UnitTestCellAverageTBB (Passed)
# Build: Ubuntu-GCC-5.4 (dejagore.kitware)
cell_average_timings = [
 1.59, #[[mean 1.46, sd = 0.84]]
 1.90, #[mean 1.59, sd = 0.85]
 2.79,
 0.81,
 2.56,
 0.82,
 1.23,
 2.24,

]
cell_average_start_mean = 1.46
cell_average_start_sd = 0.84
cell_average_end_mean = 1.76
cell_average_end_sd = 0.89
run("UnitTestCellAverageTBB", cell_average_timings, cell_average_start_mean, cell_average_start_sd)
print "cdash end mean,sd => ", (cell_average_end_mean,cell_average_end_sd)
print ""
print ""

# Test: UnitTestCudaDeviceAdapter (Passed)
# Build: Ubuntu-GCC-5.4 (dejagore.kitware)
cuda_device_timings = [
 5.19,
 8.23,
 8.60,
 41.61,
 6.70,
 9.24,
 5.90,
 4.87,
 7.74,
 7.5,
 8.41,
 43.21,
 6.73,
 9.76,
]
cuda_device_start_mean = 6.11
cuda_device_start_sd = 1.27
cuda_device_end_mean = 8
cuda_device_end_sd = 1.28

run("UnitTestCudaDeviceAdapter", cuda_device_timings, cuda_device_start_mean, cuda_device_start_sd)
print "cdash end mean,sd => ", (cuda_device_end_mean,cuda_device_end_sd)
print ""
print ""
