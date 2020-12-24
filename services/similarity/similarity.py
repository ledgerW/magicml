import sys
sys.path.append('..')
sys.setrecursionlimit(100000)

import os
import json
import pathlib
import logging
import boto3
from boto3.session import Session
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

from libs.response_lib import success, failure

# AWS X-Ray Tracing
#from aws_xray_sdk.core import xray_recorder
#from aws_xray_sdk.core import patch_all
#patch_all()

CLEAN_BUCKET = os.getenv('CLEAN_BUCKET')
MODELS_BUCKET = os.getenv('MODELS_BUCKET')
INFERENCE_BUCKET = os.getenv('INFERENCE_BUCKET')

s3 = boto3.client('s3')


def get_embeddings(event, context):
  '''
  '''
  MNT_PATH = 'opt/ml/processing'
  LOCAL_INPUT_PATH = '{}/input'.format(MNT_PATH)
  LOCAL_MODEL_PATH = '{}/model'.format(MNT_PATH)
  LOCAL_OUTPUT_PATH = '{}/output'.format(MNT_PATH)

  sagemaker_session = sagemaker.Session()
  role = 'arn:aws:iam::553371509391:role/magicml-sagemaker'
  image_uri = '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.3.1-cpu-py37-ubuntu18.04'

  model_bucket = MODELS_BUCKET
  model_prefix = 'use-large'
  model_data = 's3://{}/{}/model.tar.gz'.format(model_bucket, model_prefix)

  input_bucket = CLEAN_BUCKET
  input_prefix = 'cards'
  input_data = 's3://{}/{}/cards.csv'.format(input_bucket, input_prefix)

  output_bucket = INFERENCE_BUCKET
  output_prefix = 'use-large'
  output_data = 's3://{}/{}'.format(output_bucket, output_prefix)

  tf_processor = ScriptProcessor(
      sagemaker_session=sagemaker_session,
      role=role,
      image_uri=image_uri,
      instance_type="ml.m5.2xlarge",
      instance_count=1,
      command=['python3', '-v'],
      max_runtime_in_seconds=7200,
      base_job_name='use-large-embeddings'
  )

  tf_processor.run(
      code='src/process_embeddings.py',
      inputs=[
          ProcessingInput(
              input_name='model',
              source=model_data,
              destination=LOCAL_MODEL_PATH
          ),
          ProcessingInput(
              input_name='cards',
              source=input_data,
              destination=LOCAL_INPUT_PATH
          )
      ],
      outputs=[
          ProcessingOutput(
              output_name='embeddings',
              source=LOCAL_OUTPUT_PATH,
              destination=output_data
          )
      ],
      wait=False
  )

  return success({'status': True})


def stage_embeddings(event, context):
  MNT_PATH = os.getenv('EFS_MOUNT_PATH')
  LOCAL_INPUT_PATH = '{}/input'.format(MNT_PATH)
  LOCAL_OUTPUT_PATH = '{}/output'.format(MNT_PATH)
  CARD_EMBEDDINGS_PATH = LOCAL_INPUT_PATH + '/embeddings.csv'
  CARD_DATA_PATH = LOCAL_INPUT_PATH + '/cards.csv'
  SORTED_CARD_PATH = LOCAL_OUTPUT_PATH + '/sorted'

  os.makedirs(SORTED_CARD_PATH, exist_ok=True)

  # Get embeddings and card data from S3
  s3.download_file(OUTPUT_BUCKET, 'use-large/arena_embeddings.csv', CARD_EMBEDDINGS_PATH)
  s3.download_file(CLEAN_BUCKET, 'cards/cards.csv', CARD_DATA_PATH)

  # Get card embeddings matrix
  embed_df = pd.read_csv(CARD_EMBEDDINGS_PATH)\
    .rename(columns={'Unnamed: 0': 'Names'})

  # Get MTGJSON clean cards data
  merge_cols = [
    'Names','id','mtgArenaId','scryfallId','name','colorIdentity','colors','setName',
    'convertedManaCost','manaCost','life','loyalty','power','toughness',
    'type','types','subtypes','supertypes','text','purchaseUrls',
    'brawl','commander','duel','future','historic','legacy','modern',
    'oldschool','pauper','penny','pioneer','standard','vintage'
  ]

  cards_df = pd.read_csv(CARD_DATA_PATH)\
    .query('mtgArenaId.notnull()')\
    .assign(Names=lambda df: df.name + '-' + df.setCode)\
    .assign(Names=lambda df: df.Names.apply(lambda x: x.replace(' ', '_')))\
    [merge_cols]

  # Sort, merge, and save each card locally
  if 'n_cards' in event.keys():
    n_cards = event['n_cards']
  else:
    n_cards = len(embed_df.columns)

  for card in embed_df.columns[1:n_cards]:
    embed_df[['Names', card]]\
        .merge(cards_df, how='left', on='Names')\
        .sort_values(by=card, ascending=False)\
        .to_csv(SORTED_CARD_PATH + '/{}.csv'.format(card), index=False)

  return success({'status': True})