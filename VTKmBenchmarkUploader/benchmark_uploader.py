#!/usr/bin/env python

"""Utility script to upload a collection of benchmark files to an s3 bucket


By default we use what ever aws credintials you have set up, but you can
explicitly specify an aws credential file with '--credentials' and a specific
profile from that credential with '--profile'


# > benchmark_uploader.py  <directory> <bucket> --today
"""

from __future__ import print_function
import argparse
import boto3
import datetime
import hashlib
import os
import re
import string
import sys
import time

from botocore.exceptions import ClientError


# Construct a boto session using the provided credentials and profile
def construct_session(credentials, profile):
  if credentials:
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = credentials

  return boto3.Session(profile_name=profile)

# Find all files in a directory which have match a regex pattern
def collect_files_by_pattern(directory, pattern):
  result = []
  cached_pattern = re.compile(pattern)

  for file in os.listdir(directory):
    full_path = os.path.join(directory, file)
    matches = cached_pattern.match(file)
    if matches and os.path.isfile(full_path):
      result.append( os.path.join(directory, file) )

  return result

# Find all files in a directory which have been modified in the last 24h
def collect_files_by_timestamp(directory):
  result = []
  window = datetime.timedelta(minutes=1440)
  oldest_time_allowed = datetime.datetime.now() - window

  for file in os.listdir(directory):
    full_path = os.path.join(directory, file)
    f_info = os.stat(full_path)
    f_time = datetime.datetime.fromtimestamp(f_info.st_mtime)
    if f_time >= oldest_time_allowed:
      result.append( full_path )
  return result


def exit_bad_search_parameters():
  print("Was provided invalid pattern and timestamp information so unable to run")
  print("This is not expected")
  exit(2)

def exit_wrong_bucket_name(name, names):
  print("The aws s3 doesn't contain a bucket named: %s" %(name))
  print("The possible buckets are: ")
  for n in names:
    print("\t",n)
  exit(2)

def exit_failed_s3_connection(err):
  print(err)
  exit(2)


def bucket_has_key(client, bucket, key):
  try:
    obj = client.head_object(Bucket=bucket, Key=key)
    return True
  except ClientError as exc:
      if exc.response['Error']['Code'] != '404':
          return False

def main(session, bucket_name, directory, pattern, by_timestamp):
  print('session', session)
  print('bucket', bucket_name)
  print('directory', directory)
  print('pattern', pattern)
  print('by_timestamp', by_timestamp)

  files = []
  if pattern and by_timestamp:
    pfiles = collect_files_by_pattern(directory, pattern)
    tfiles = collect_files_by_pattern(directory, pattern)
    files = set.intersection( set(pfiles), set(tfiles) )
    pass
  elif pattern:
    files = collect_files_by_pattern(directory, pattern)
  elif by_timestamp:
    files = collect_files_by_timestamp(directory)
  else:
    exit_bad_search_parameters()

  try:
    client = session.client('s3')
    response = client.list_buckets()
  except Exception as err:
    exit_failed_s3_connection(err)

  bucket_names = [b['Name'] for b in response['Buckets']]

  if not bucket_name in bucket_names:
    exit_wrong_bucket_name(bucket_name, bucket_names)

  for file in files:
    f_name = os.path.basename(file)
    if not bucket_has_key(client, bucket_name, f_name):
      sha256value = hashlib.sha256(open(file,'rb').read()).hexdigest()

      print("uploading '%s' to bucket '%s' with hash %s" %(f_name, bucket_name, sha256value))
      client.upload_file(file, bucket_name, f_name,
        ExtraArgs={"Metadata": {"sha256": sha256value}}
        )
    else:
      print("'%s' has already been uploaded to bucket '%s'" %(f_name, bucket_name))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Upload a collection of benchmarks.')
  parser.add_argument('directory', nargs=1, help='directory you want searched for files')
  parser.add_argument('bucket', nargs=1, help='specify the s3 bucket to upload too')
  parser.add_argument('-p', '--pattern',nargs=1, help='only upload all files that match the given pattern (default: .*)')
  parser.add_argument('--today', action='store_true', help='only upload files that are created today')
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
  if not pattern and not today:
    pattern = '.*'

  #start the upload process
  main(session, bucket, directory, pattern, today)

