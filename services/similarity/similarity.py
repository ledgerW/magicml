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
import tensorflow as tf
import tensorflow_hub as hub
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
MODELS_BUCKET = os.getenv('MODELS_BUCKET')
INFERENCE_BUCKET = os.getenv('INFERENCE_BUCKET')
LOCAL_RAW_PATH = '{}/raw'.format(MNT_PATH)
LOCAL_CLEAN_PATH = '{}/clean'.format(MNT_PATH)
LOCAL_MODEL_PATH = '{}/model'.format(MNT_PATH)
LOCAL_SIM_PATH = '{}/similarity'.format(MNT_PATH)

s3 = boto3.client('s3')


def get_embeddings(event, context):
    '''
    '''
    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

    # Load mode from S3 if not already in EFS
    if not os.path.exists(LOCAL_MODEL_PATH + '/use-large'):
      res = s3.list_objects_v2(
          Bucket=MODELS_BUCKET,
          Prefix='use-large'
      )

      model_files = [file['Key'] for file in res['Contents']]

      # Get raw data from S3
      for s3_key in model_files:
        local_path = LOCAL_MODEL_PATH + '/use-large/' + s3_key
        os.makedirs(pathlib.Path(local_path).parent, exist_ok=True)
        s3.download_file(MODELS_BUCKET, s3_key, local_path)


    # Get all embeddings
    cards_df = pd.read_csv(LOCAL_CLEAN_PATH + '/cards.csv')

    arena_df = cards_df.query('mtgArenaId.notnull()')\
      .reset_index(drop=True)\
      .fillna(value={'text': 'Blank'})

    arena_txt = list(arena_df.text)
    arena_name = [(name + '-' + set_name).replace(' ','_') for name, set_name in zip(arena_df.name, arena_df.setCode)]

    use_embed = hub.KerasLayer(LOCAL_MODEL_PATH + '/use-large')

    embeddings = use_embed(arena_txt)

    corr = np.inner(embeddings, embeddings)

    pd.DataFrame(corr, columns=arena_name, index=arena_name)\
      .to_csv(LOCAL_SIM_PATH + '/use_large_cards.csv')
    
    # Send embeddings to S3
    for local_path in pathlib.Path(LOCAL_SIM_PATH).glob('*'):
      s3.upload_file(str(local_path), INFERENCE_BUCKET, 'use-large/{}'.format(local_path.name))

    return success({'status': True})