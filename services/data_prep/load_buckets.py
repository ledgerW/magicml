import sys
sys.path.append('..')
sys.setrecursionlimit(100000)

import os
import json
import pathlib
import logging
import boto3
import numpy as np
import pandas as pd
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from libs.response_lib import success, failure

# AWS X-Ray Tracing
#from aws_xray_sdk.core import xray_recorder
#from aws_xray_sdk.core import patch_all
#patch_all()

MNT_PATH = os.getenv('EFS_MOUNT_PATH')
RAW_BUCKET = os.getenv('RAW_BUCKET')
CLEAN_BUCKET = os.getenv('CLEAN_BUCKET')
LOCAL_RAW_PATH = '{}/raw'.format(MNT_PATH)
LOCAL_CLEAN_PATH = '{}/clean'.format(MNT_PATH)

s3 = boto3.client('s3')


def get_raw_data(event, context):
    '''
    '''
    os.makedirs(LOCAL_CLEAN_PATH, exist_ok=True)
    os.makedirs(LOCAL_RAW_PATH, exist_ok=True)

    res = s3.list_objects_v2(
        Bucket=RAW_BUCKET,
        Prefix='mtgjson'
    )

    card_files = [file['Key'] for file in res['Contents'] if '/decks/' not in file['Key']]

    # Get raw data from S3
    for s3_key in card_files:
      local_path = LOCAL_RAW_PATH + '/' + s3_key.split('/')[-1]
      s3.download_file(RAW_BUCKET, s3_key, local_path)

    # Get raw data from MTGJSON
    
    
    # Send raw data to S3
    for local_path in pathlib.Path(LOCAL_CLEAN_PATH).glob('*'):
      s3.upload_file(str(local_path), CLEAN_BUCKET, 'cards/{}'.format(local_path.name))

    return success({'status': True})


def get_pretrained_model(event, context):
    '''
    '''
    os.makedirs(LOCAL_CLEAN_PATH, exist_ok=True)
    os.makedirs(LOCAL_RAW_PATH, exist_ok=True)

    res = s3.list_objects_v2(
        Bucket=RAW_BUCKET,
        Prefix='mtgjson'
    )

    card_files = [file['Key'] for file in res['Contents'] if '/decks/' not in file['Key']]

    # Get raw data from S3
    for s3_key in card_files:
      local_path = LOCAL_RAW_PATH + '/' + s3_key.split('/')[-1]
      s3.download_file(RAW_BUCKET, s3_key, local_path)

    # Get model (USE-large) from TF Hub
    
    
    # Send raw data to S3
    for local_path in pathlib.Path(LOCAL_CLEAN_PATH).glob('*'):
      s3.upload_file(str(local_path), CLEAN_BUCKET, 'cards/{}'.format(local_path.name))

    return success({'status': True})