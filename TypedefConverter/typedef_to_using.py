#!/usr/bin/env python

"""Utility script to convert C++ typedef to using statement without compilation.
This is done as versions such as clang-tidy replace the typedef with the
first explicit instantiation. Plus by doing this as just a parser, we don't
have to worry about writing code that will use all the existing typedefs

To run the script, invoke

# > typedef_to_using.py <directory>
"""


import re
import sys
import os
import string


extensions=[".h", ".hxx", ".cxx", ".cu", ".cpp"]


def single_line_conversion(line):
  """ converts a single line typedef to a using statement, preserving whitespace
  """
  components = re.split('\s', line)
  name = components[-1][:-1] #last item, drop the trailing space

  typedef_index = 0
  for c in components:
    if c == 'typedef':
      break
    typedef_index = typedef_index + 1

  using = 'using ' + name + ' ='
  components[typedef_index] = using

  new_line = ' '.join(components[:-1]) + ';'
  return new_line



def multiple_line_conversion(lines):
  """ converts a multiple line typedef to a using statement, preserving whitespace
  """
  #we know for a fact that we only care about the first, and last line!
  last_line_components = re.split('\s', lines[-1])
  name = last_line_components[-1][:-1] #last item, drop the trailing space



  #simulate a first line to reuse the typedef -> using logic
  first_line =  single_line_conversion(lines[0]+ ' ' + name +';')
  first_line = first_line[:-1]

  #now we need to remove the name from the last line
  last_line = ' '.join(last_line_components[:-1]) + ';'

  new_lines = lines
  new_lines[0] = first_line
  new_lines[-1] = last_line

  return new_lines


def process(file):
  filename, file_extension = os.path.splitext(file)
  if file_extension not in extensions:
    return

  with open(file, 'r') as f:
    contents = f.read()
    lines = contents.split('\n')
    f.close()

  out_file = open(file, 'w')
  iterator = iter(lines[:-1])
  while iterator:
    try:
      line = iterator.next()
    except StopIteration:
      break
    has_typedef = re.match('\s*typedef', line)
    has_semicolon = re.match('.*;$', line)
    if has_typedef and has_semicolon:
      line = single_line_conversion(line)
      out_file.write('%s\n' % line)
    elif has_typedef:
      #we need to iterate forward in lines capturing all lines intill
      #we capture a semicolon
      mult_lines = [line]
      while not has_semicolon:
        line = iterator.next()
        mult_lines.append(line)
        has_semicolon = re.match('.*;$', line)

      #at this point we need to convert multiple lines
      mult_lines = multiple_line_conversion(mult_lines)
      for elem in mult_lines:
        out_file.write('%s\n' % elem)
    else:
      out_file.write('%s\n' % line)
  out_file.close()

if __name__ == "__main__":
  for root, dirs, files in os.walk(sys.argv[1]):
    path = root.split(os.sep)
    for file in files:
      process(file)

