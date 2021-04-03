import sys
sys.path.append('..')
sys.path.append('../..')
sys.setrecursionlimit(100000)

import os
import json
import boto3
from boto3.dynamodb.conditions import Key
import libs.dynamodb_lib as dynamodb_lib
from libs.response_lib import success, failure

# Load only if in Docker Lambda Environment
if os.getenv('CONTAINER_ENV'):
  import numpy as np
  import pandas as pd
  import tensorflow as tf

# AWS X-Ray Tracing
#from aws_xray_sdk.core import xray_recorder
#from aws_xray_sdk.core import patch_all
#patch_all()

STAGE = os.getenv('STAGE')
CLEAN_BUCKET = os.getenv('CLEAN_BUCKET')
MODELS_BUCKET = os.getenv('MODELS_BUCKET')
SRC_BUCKET = os.getenv('SOURCE_BUCKET')
INFERENCE_BUCKET = os.getenv('INFERENCE_BUCKET')
SIMILARITY_TABLE = os.getenv('SIMILARITY_TABLE')
MNT_PATH = os.getenv('EFS_MOUNT_PATH')
LOCAL_INPUT_PATH = '{}/input'.format(MNT_PATH)
EMBEDDINGS_PATH = LOCAL_INPUT_PATH + '/embeddings.npy'
CARD_DATA_PATH = LOCAL_INPUT_PATH + '/cards.csv'


def card_query(event, context):
  '''
  event['key']: what index/attribute to search by
  event['value']: the value of the search attribute
  '''
  print(event)

  try:
    key = event['key']
    value = event['value'].replace('__','//')
  except:
    key = json.loads(event['body'])['key']
    value = json.loads(event['body'])['value'].replace('__','//')

  params = {
    'Item': Key(key).eq(value)
  }
  cards = dynamodb_lib.call(SIMILARITY_TABLE, 'query', params)
  cards = cards['Items']
  
  return success({'cards': cards})


def free_text_query(event, context):
  '''
  event['key']: what index/attribute to search by
  event['value']: the value of the search attribute
  '''
  print(event)
  try:
    query = event['query']
  except:
    query = json.loads(event['body'])['query']

  # Load model
  use_embed = tf.saved_model.load('models/use-large/1')

  # Load card embeddings and names
  all_embeds = np.load(EMBEDDINGS_PATH)

  merge_cols = [
    'Names','id','mtgArenaId','scryfallId','name','colors','setName',
    'convertedManaCost','manaCost','loyalty','power','toughness',
    'type','types','subtypes','text','image_urls',
    'brawl','commander','duel','future','historic','legacy','modern',
    'oldschool','pauper','penny','pioneer','standard','vintage'
  ]
  cards_df = pd.read_csv('cardsS3.csv')\
    .assign(Names=lambda df: df.name + '-' + df.id.astype('str'))\
    .assign(Names=lambda df: df.Names.apply(lambda x: x.replace(' ', '_').replace('//', 'II')))\
    .fillna('0')\
    [merge_cols]

  card_names = [
  (name + '-' + str(id_val)).replace(' ','_').replace('//', 'II') for name, id_val in zip(cards_df.name, cards_df.id)
  ]

  # Get query embedding
  embed_query = use_embed(query)

  # Get similar cards
  sims = np.inner(all_embeds, embed_query)

  sims_list = pd.DataFrame(sims, columns=['free_text_query'], index=cards_name)\
    .sort_values(by='free_text_query', ascending=False)\
    .head(50)\
    .reset_index()\
    .rename(columns={'index':'Names', 'free_text_query':'similarity'})\
    .merge(cards_df, how='left', on='Names')\
    .assign(similarity=lambda df: df.similarity.astype('str'))\
    .assign(id=lambda df: df.id.astype('str'))\
    .assign(mtgArenaId=lambda df: df.mtgArenaId.astype('str'))\
    .assign(loyalty=lambda df: df.loyalty.astype('str'))\
    .assign(power=lambda df: df.power.astype('str'))\
    .assign(toughness=lambda df: df.toughness.astype('str'))\
    .assign(convertedManaCost=lambda df: df.convertedManaCost.astype('str'))\
    .to_dict(orient='records')

  cards = [{
      'freeText': query,
      'similarities': sims_list
  }]
  
  return success({'cards': cards})


