#!/usr/bin/env python

"""Utility script to upload a collection of benchmark files to an s3 bucket


By default we use what ever aws credintials you have set up, but you can
explicitly specify an aws credential file with '--credentials' and a specific
profile from that credential with '--profile'


# > benchmark_uploader.py  <directory> <bucket> --today
"""

from __future__ import print_function
import argparse
import datetime
import os
import re
import string
import boto3
import sys
import time


def construct_session(credentials, profile):
  if credentials:
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = credentials

  return boto3.Session(profile_name=profile)

def upload(session, bucket, files):
  pass

def collect_files_by_pattern(directory, pattern):
  pass

def collect_files_by_timestamp(directory):
  pass


def wrong_bucket_error_message(name, names):
  print("The aws s3 doesn't contain a bucket named: %s" %(names))
  print("The possible buckets are: ")
  for n in names:
    print("\t",n)
  exit(2)

def failed_s3_connection(err):
  print(err)
  exit(2)


def main(session, bucket_name, directory, pattern, by_timestamp):
  print('session', session)
  print('bucket', bucket)
  print('directory', directory)
  print('pattern', pattern)
  print('by_timestamp', by_timestamp)

  files = []
  if pattern:
    files = files + collect_files_by_pattern(directory, pattern)

  if by_timestamp:
    files = files + collect_files_by_timestamp(directory)

  try:
    client = session.client('s3')
    response = client.list_buckets()
  except Exception as err:
    failed_s3_connection(err)

  bucket_names = [b['Name'] for b in response['Buckets']]
  has_bucket = bucket_name in bucket_names

  if not has_bucket:
    wrong_bucket_error_message(bucket_name, bucket_names)
    exit(2)

  for file in files:
    f_name = os.path.basename(file)
    print(f_name)
  #   print("client.upload_file(%s, %s, %s)" %(file,bucket_name,f_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Upload a collection of benchmarks.')
  parser.add_argument('directory', nargs=1, help='directory you want searched for files')
  parser.add_argument('bucket', nargs=1, help='specify the s3 bucket to upload too')
  parser.add_argument('-p', '--pattern', nargs=1, help='upload all files that match the given pattern')
  parser.add_argument('-t', '--today', action='store_true', help='upload all files that are created today')
  parser.add_argument('--credentials',  nargs=1,  help='aws credential file to use')
  parser.add_argument('--profile',  nargs=1,  help='aws credential profile name to use')
  args = parser.parse_args()

  #setup session
  credentials = args.credentials[0] if args.credentials else None
  profile = args.profile[0] if args.profile else None
  session = construct_session(credentials, profile)

  #setup variables

  bucket = args.bucket[0]
  directory = args.directory[0]
  pattern = args.pattern[0] if args.pattern else None
  today = args.today

  #start the upload process
  main(session, bucket, directory, pattern, today)

