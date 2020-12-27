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


def get_image_uris(row):
    try:
        if pd.notna(row['image_uris']):
            return row['image_uris']
        else:
            return [card['image_uris'] for card in row['card_faces']]
    except:
        return 'Blank'


def worker(event, context):
    '''
    '''
    os.makedirs(LOCAL_CLEAN_PATH, exist_ok=True)
    os.makedirs(LOCAL_RAW_PATH, exist_ok=True)

    # Get MTGJSON data
    res = s3.list_objects_v2(
        Bucket=RAW_BUCKET,
        Prefix='mtgjson'
    )
    card_files = [file['Key'] for file in res['Contents'] if '/decks/' not in file['Key']]
    for s3_key in card_files:
      local_path = LOCAL_RAW_PATH + '/' + s3_key.split('/')[-1]
      s3.download_file(RAW_BUCKET, s3_key, local_path)

    # Get Scryfall data
    res = s3.list_objects_v2(
        Bucket=RAW_BUCKET,
        Prefix='scryfall'
    )
    card_files = [file['Key'] for file in res['Contents'] if 'cards.json' in file['Key']]
    for s3_key in card_files:
      local_path = LOCAL_RAW_PATH + '/' + 'scryfall_cards.json'
      s3.download_file(RAW_BUCKET, s3_key, local_path)

    # Clean data
    cards_df = pd.read_csv(LOCAL_RAW_PATH + '/cards.csv')\
      .drop(columns=['index'])

    sets_df = pd.read_csv(LOCAL_RAW_PATH + '/sets.csv')[['code','name']]\
      .rename(columns={'name': 'setName', 'code':'setCode'})

    cards_df = cards_df\
      .merge(sets_df, how='left', on='setCode')

    legs_df = pd.read_csv(LOCAL_RAW_PATH + '/legalities.csv')\
      .pivot(index='uuid', columns='format', values='status')\
      .reset_index()\
      .fillna('Blank')

    cards_df = cards_df\
      .merge(legs_df, how='left', on='uuid')

    scryfall_df = pd.read_json(LOCAL_RAW_PATH + '/scryfall_cards.json')\
        .query('arena_id.notnull()')\
        [['id','image_uris','card_faces']]\
        .reset_index(drop=True)\
        .assign(image_urls=lambda df: df.apply(get_image_uris, axis=1))\
        .drop(columns=['image_uris','card_faces'])\
        .rename(columns={'id': 'scryfallId'})

    cards_df = cards_df\
      .merge(scryfall_df, how='left', on='scryfallId')

    cards_df.to_csv(LOCAL_CLEAN_PATH + '/cards.csv', index=False)
    
    # Send clean data to S3
    for local_path in pathlib.Path(LOCAL_CLEAN_PATH).glob('*'):
      s3.upload_file(str(local_path), CLEAN_BUCKET, 'cards/{}'.format(local_path.name))

    return success({'status': True})