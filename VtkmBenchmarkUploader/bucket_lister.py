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


# Construct a boto session using the provided credentials and profile
def construct_session(credentials, profile):
  if credentials:
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = credentials

  return boto3.Session(profile_name=profile)

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

def main(session, bucket_name, download, directory):
  print('session', session)
  print('bucket', bucket_name)
  print('download', download)

  try:
    client = session.client('s3')
    response = client.list_buckets()
  except Exception as err:
    exit_failed_s3_connection(err)

  bucket_names = [b['Name'] for b in response['Buckets']]

  if not bucket_name in bucket_names:
    exit_wrong_bucket_name(bucket_name, bucket_names)

  #now to actually list / download all files
  objs = client.list_objects_v2(Bucket=bucket_name)['Contents']
  print("Files in the bucket: %s are: " %(bucket_name))
  for object in objs:
    key = object['Key']
    print("\t %s" %(key))
    if download:
      fpath = os.path.join(directory, key)
      client.download_file(bucket_name, key, fpath)


  #client.delete_object(Bucket=bucket_name, Key="")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='list all the keys in a bucket and optional download the objects related to those keys.')
  parser.add_argument('bucket', nargs=1, help='specify the s3 bucket to list/download')
  parser.add_argument('-d', '--download', action='store_true', help='download all files in the bucket')
  parser.add_argument('-o', '--directory', nargs=1, help='directory to download files too ( default: cwd )')
  parser.add_argument('--credentials',  nargs=1,  help='aws credential file to use')
  parser.add_argument('--profile',  nargs=1,  help='aws credential profile name to use')
  args = parser.parse_args()

  #setup session
  credentials = args.credentials[0] if args.credentials else None
  profile = args.profile[0] if args.profile else None
  session = construct_session(credentials, profile)

  #setup variables
  bucket = args.bucket[0]
  download = args.download
  directory = args.directory[0] if args.directory else os.getcwd()

  #start the upload process
  main(session, bucket, download, directory)

