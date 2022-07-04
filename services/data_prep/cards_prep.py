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
      print(s3_key)
      local_path = LOCAL_RAW_PATH + '/' + s3_key.split('/')[-1]
      s3.download_file(RAW_BUCKET, s3_key, local_path)

    # Get Scryfall data
    res = s3.list_objects_v2(
        Bucket=RAW_BUCKET,
        Prefix='scryfall'
    )
    card_files = [file['Key'] for file in res['Contents'] if 'cards.json' in file['Key']]
    for s3_key in card_files:
      print(s3_key)
      local_path = LOCAL_RAW_PATH + '/' + 'scryfall_cards.json'
      s3.download_file(RAW_BUCKET, s3_key, local_path)

    # Get Supported Sets file
    res = s3.list_objects_v2(
        Bucket=RAW_BUCKET,
        Prefix='supported_sets'
    )
    card_files = [file['Key'] for file in res['Contents'] if 'supported_sets.txt' in file['Key']]
    for s3_key in card_files:
      print(s3_key)
      local_path = LOCAL_RAW_PATH + '/' + 'supported_sets.txt'
      s3.download_file(RAW_BUCKET, s3_key, local_path)

    supported_sets = list(pd.read_csv(LOCAL_RAW_PATH + '/supported_sets.txt', sep='\n', names=['sets']).sets)

    # Prep data
    # Get MTGJSON data
    cards_df = pd.read_csv(LOCAL_RAW_PATH + '/cards.csv')\
      .drop(columns=['index'])

    # Merge with sets data
    sets_df = pd.read_csv(LOCAL_RAW_PATH + '/sets.csv')[['code','name']]\
      .rename(columns={'name': 'setName', 'code':'setCode'})

    cards_df = cards_df\
      .merge(sets_df, how='left', on='setCode')

    # Merge with legalities / formats data
    legs_df = pd.read_csv(LOCAL_RAW_PATH + '/legalities.csv')\
      .drop_duplicates(subset=['uuid','format','status'])\
      .pivot(index='uuid', columns='format', values='status')\
      .reset_index()\
      .fillna('Blank')

    cards_df = cards_df\
      .merge(legs_df, how='left', on='uuid')

    # Merge with scryfall data
    scryfall_df = pd.read_json(LOCAL_RAW_PATH + '/scryfall_cards.json')
    scryfall_sets = scryfall_df.set_name.unique()
    supported_sets_varients = [scry_s for scry_s in scryfall_sets if any([s in scry_s for s in supported_sets])]
    print(supported_sets_varients)

    scryfall_df = scryfall_df\
        .query('set_name == @supported_sets_varients')\
        [['id','image_uris','card_faces']]\
        .reset_index(drop=True)\
        .assign(image_urls=lambda df: df.apply(get_image_uris, axis=1))\
        .drop(columns=['image_uris','card_faces'])\
        .rename(columns={'id': 'scryfallId'})

    cards_df = cards_df\
      .merge(scryfall_df, how='left', on='scryfallId')

    # Save cards for NLP in local EFS
    cards_df\
      .query('setName == @supported_sets_varients')\
      .to_csv(LOCAL_CLEAN_PATH + '/cards.csv', index=False)

    # Save cards in BeIR format for GPL Domain Adaptation
    cards_df\
      .query('setName == @supported_sets_varients')\
      .query('text.notnull()')\
      .assign(title='')\
      .assign(id=lambda df: df.id.astype(str))\
      .rename(columns={'id':'_id'})\
      [['text','title','_id']]\
      .to_json(LOCAL_CLEAN_PATH + '/corpus.jsonl', orient='records', lines=True)

    # Save ALL cards in local EFS
    cards_df\
      .to_csv(LOCAL_CLEAN_PATH + '/all_cards.csv', index=False)
    
    # Send clean data to S3
    for local_path in pathlib.Path(LOCAL_CLEAN_PATH).glob('*'):
      s3.upload_file(str(local_path), CLEAN_BUCKET, 'cards/{}'.format(local_path.name))

    return success({'status': True})