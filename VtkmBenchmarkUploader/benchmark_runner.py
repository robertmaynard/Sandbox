#!/usr/bin/env python

"""Utility script to run a collection of executable/benchmarks and
save the output of the executables to files.


The output files the script generate will have the following name:
  executable_hostname_yy_mm_dd

Where executable is resolved to the basename of the executable
Where hostname is the first word of fully qualified domain name when split
on '.'.

So for example running the benchmark BenchmarkReduce on the machine
'bigboard.website.org` on Oct 1 2017 will generate the file:
  BenchmarkReduce_bigboard_17_10_01

# > benchmark_runner.py -d <directory> -p <executable_pattern>
"""

from __future__ import print_function
import argparse
import datetime
import os
import re
import socket
import subprocess
import sys
import time


# Given a directory, a pattern ( used only on the file name) and if we
# want to recursively walk return a list of executables
def collect_executables(directory, recursive, pattern):
  result = []
  cached_pattern = re.compile(pattern)

  def is_valid(dir, file, pattern):
    full_path = os.path.join(dir, file)
    is_exec = os.path.isfile(full_path) and os.access(full_path, os.X_OK)
    matches = cached_pattern.match(file)
    return matches and is_exec

  if recursive:
    for root, dirs, files in os.walk(directory):
      for file in files:
        if is_valid(root, file, pattern):
          result.append( os.path.join(root, file) )
  else:
    for file in os.listdir(directory):
      if is_valid(directory, file, pattern):
        result.append( os.path.join(directory, file) )

  return result


# Given an absolute path to an executable and a dir
def process(execut, output_directory):

  #construct a name using the format:
  #  executable_hostname_yy_mm_dd
  out_name = os.path.basename(execut)
  utcnow = datetime.datetime.utcnow()
  fqdn = socket.getfqdn().split('.')[0]

  out_name += "_"
  out_name += fqdn
  out_name += str(utcnow.strftime("_%y_%m_%d"))

  exec_dir = os.path.dirname(execut)
  output_path = os.path.join(output_directory,out_name)
  output_file = open(output_path, 'w')
  args = [execut]

  print('running', execut, 'without output saved to', output_path)
  process = subprocess.Popen(args,
                             bufsize=4096,
                             stdout=output_file,
                             stderr=subprocess.STDOUT,
                             shell=False,
                             cwd=exec_dir)
  while process.poll() is None:
    print('.', end='')
    sys.stdout.flush()
    time.sleep(0.5)
  process.wait()
  print(execut, 'finished')

def main(input_directories, recursive, output_directory, pattern):
  if not isinstance(input_directories, list):
    input_directories = [input_directories]

  #build up the list of executables to run
  for idir in input_directories:
    print('searching', idir, 'for executables.' )
    files = collect_executables(idir, recursive, pattern)

  #now run each executable
  for execut in files:
    process(execut, output_directory)

if __name__ == '__main__':

  loc = [os.getcwd()]
  pattern = ['.*']

  parser = argparse.ArgumentParser(description='Run a collection of benchmarks.')
  parser.add_argument('-d', '--directory', nargs='*',  default=loc, help='directories you want walked (default: current directory)')
  parser.add_argument('-p', '--pattern', nargs=1, default=pattern, help='regex pattern to use to find executables to run (default: .*)')
  parser.add_argument('-o', '--out-directory', nargs=1, default=loc, help='directory to place output files (default: current directory)')
  parser.add_argument('-r', '--recursive', action='store_true', help='recursively walk the input directories')


  args = parser.parse_args()
  main(args.directory, args.recursive, args.out_directory[0], args.pattern[0])

