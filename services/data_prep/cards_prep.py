import sys
sys.path.append('..')
sys.setrecursionlimit(100000)

import os
import json
import logging
import boto3
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from libs.response_lib import success, failure

# AWS X-Ray Tracing
#from aws_xray_sdk.core import xray_recorder
#from aws_xray_sdk.core import patch_all
#patch_all()

MNT_PATH = os.getenv('EFS_MOUNT_PATH')
RAW_BUCKET = os.getenv['RAW_BUCKET']
CLEAN_BUCKET = os.getenv['CLEAN_BUCKET']
LOCAL_CLEAN_PATH = '{}/clean'.format(MNT_PATH)

s3 = boto3.client('s3')


def worker(event, context):
    '''
    '''
    print(event)

    os.makedirs(LOCAL_CLEAN_PATH, exist_ok=True)

    body = json.loads(event['body'])

    # Get raw data from S3
    for file_key in card_files:
      print(file_key)
      s3.download_file(RAW_BUCKET, file_key, MNT_PATH + '/')

    # Clean data
    # DO THAT HERE

    # Send clean data to S3
    s3.upload_file(local_path, bucket, file_key)

    return success({'status': True})