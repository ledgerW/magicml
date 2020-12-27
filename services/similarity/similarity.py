import sys
sys.path.append('..')
sys.setrecursionlimit(100000)

import os
import json
from decimal import Decimal
import pathlib
import datetime
from time import sleep
import boto3
import pandas as pd

import libs.dynamodb_lib as dynamodb_lib
from libs.response_lib import success, failure

# AWS X-Ray Tracing
#from aws_xray_sdk.core import xray_recorder
#from aws_xray_sdk.core import patch_all
#patch_all()

STAGE = os.getenv('STAGE')
CLEAN_BUCKET = os.getenv('CLEAN_BUCKET')
MODELS_BUCKET = os.getenv('MODELS_BUCKET')
SRC_BUCKET = os.getenv('SOURCE_BUCKET')
INFERENCE_BUCKET = os.getenv('INFERENCE_BUCKET')
SM_ROLE = os.getenv('SM_ROLE')
SIMILARITY_TABLE = os.getenv('SIMILARITY_TABLE')

s3 = boto3.client('s3')
sm = boto3.client('sagemaker')
lambda_client = boto3.client('lambda')


def get_embeddings(event, context):
  '''
  '''
  MNT_PATH = '/opt/ml/processing'
  LOCAL_INPUT_PATH = '{}/input'.format(MNT_PATH)
  LOCAL_MODEL_PATH = '{}/model'.format(MNT_PATH)
  LOCAL_CODE_PATH = '{}/input/code'.format(MNT_PATH)
  LOCAL_OUTPUT_PATH = '{}/output'.format(MNT_PATH)

  image_uri = '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.3.1-cpu-py37-ubuntu18.04'

  model_prefix = 'use-large'
  model_data = 's3://{}/{}/model.tar.gz'.format(MODELS_BUCKET, model_prefix)

  input_prefix = 'cards'
  input_data = 's3://{}/{}/cards.csv'.format(CLEAN_BUCKET, input_prefix)

  src_prefix = 'sm_processing'
  src_code = 's3://{}/{}/process_embeddings.py'.format(SRC_BUCKET, src_prefix)

  output_prefix = 'use-large'
  output_data = 's3://{}/{}'.format(INFERENCE_BUCKET, output_prefix)

  s3.upload_file('src/process_embeddings.py', SRC_BUCKET, 'sm_processing/process_embeddings.py')

  now = datetime.datetime.now().strftime(format='%Y-%d-%m-%H-%M-%S')

  sm.create_processing_job(
    ProcessingJobName='use-large-embeddings-{}'.format(now),
    RoleArn=SM_ROLE,
    StoppingCondition={
        'MaxRuntimeInSeconds': 7200
    },
    AppSpecification={
        'ImageUri': image_uri,
        'ContainerEntrypoint': [
            'python3',
            '-v',
            (LOCAL_CODE_PATH + '/process_embeddings.py')
        ]
    },
    ProcessingResources={
        'ClusterConfig': {
            'InstanceCount': 1,
            'InstanceType': 'ml.m5.2xlarge',
            'VolumeSizeInGB': 30
        }
    },
    ProcessingInputs=[
        {
            'InputName': 'model',
            'S3Input': {
                'S3Uri': model_data,
                'LocalPath': LOCAL_MODEL_PATH,
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
                'S3DataDistributionType': 'FullyReplicated'
            }
        },
        {
            'InputName': 'cards',
            'S3Input': {
                'S3Uri': input_data,
                'LocalPath': LOCAL_INPUT_PATH,
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
                'S3DataDistributionType': 'FullyReplicated',
            }
        },
        {
            'InputName': 'code',
            'S3Input': {
                'S3Uri': src_code,
                'LocalPath': LOCAL_CODE_PATH,
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
                'S3DataDistributionType': 'FullyReplicated'
            }
        }
    ],
    ProcessingOutputConfig={
        'Outputs': [
            {
                'OutputName': 'embeddings',
                'S3Output': {
                    'S3Uri': output_data,
                    'LocalPath': LOCAL_OUTPUT_PATH,
                    'S3UploadMode': 'EndOfJob'
                }
            }
        ]
    }
  )

  return success({'status': True})


def stage_embed_master(event, context):
  '''
  event['n_cards']: number of cards to batch and send to worker
  event['batch_size']: number of cards for each worker to handle
  '''
  MNT_PATH = os.getenv('EFS_MOUNT_PATH')
  LOCAL_INPUT_PATH = '{}/input'.format(MNT_PATH)
  LOCAL_OUTPUT_PATH = '{}/output'.format(MNT_PATH)
  CARD_EMBEDDINGS_PATH = LOCAL_INPUT_PATH + '/embeddings.csv'
  CARD_DATA_PATH = LOCAL_INPUT_PATH + '/cards.csv'
  SORTED_CARD_PATH = LOCAL_OUTPUT_PATH + '/sorted'

  os.makedirs(LOCAL_INPUT_PATH, exist_ok=True)
  os.makedirs(LOCAL_OUTPUT_PATH, exist_ok=True)
  os.makedirs(SORTED_CARD_PATH, exist_ok=True)

  # Get embeddings and card data from S3
  if not os.path.exists(CARD_EMBEDDINGS_PATH):
    s3.download_file(INFERENCE_BUCKET, 'use-large/arena_embeddings.csv', CARD_EMBEDDINGS_PATH)
  if not os.path.exists(CARD_DATA_PATH):
    s3.download_file(CLEAN_BUCKET, 'cards/cards.csv', CARD_DATA_PATH)

  # Get card embeddings matrix
  all_cards = pd.read_csv(CARD_EMBEDDINGS_PATH)\
    .rename(columns={'Unnamed: 0': 'Names'})\
    .columns

  if event['n_cards'] > 0:
    n_cards = event['n_cards']
    all_cards = all_cards[0:n_cards]

  BATCH_SIZE = event['batch_size']
  batches = [list(all_cards[n:n+BATCH_SIZE]) for n in range(1, len(all_cards), BATCH_SIZE)]

  # Sort, merge, and save each card locally
  for cards in batches:
    payload = {'cards': cards}

    res = lambda_client.invoke(
        FunctionName='magicml-similarity-{}-stage_embed_worker'.format(STAGE),
        InvocationType='Event',
        Payload=json.dumps(payload)
    )

  return success({'status': True})


def stage_embed_worker(event, context):
  '''
  event['cards']: list of card names to handle
  '''
  MNT_PATH = os.getenv('EFS_MOUNT_PATH')
  LOCAL_INPUT_PATH = '{}/input'.format(MNT_PATH)
  LOCAL_OUTPUT_PATH = '{}/output'.format(MNT_PATH)
  CARD_EMBEDDINGS_PATH = LOCAL_INPUT_PATH + '/embeddings.csv'
  CARD_DATA_PATH = LOCAL_INPUT_PATH + '/cards.csv'
  SORTED_CARD_PATH = LOCAL_OUTPUT_PATH + '/sorted'

  # Get card embeddings matrix
  embed_df = pd.read_csv(CARD_EMBEDDINGS_PATH)\
    .rename(columns={'Unnamed: 0': 'Names'})

  # Get MTGJSON clean cards data
  merge_cols = [
    'Names','id','mtgArenaId','scryfallId','name','colors','setName',
    'convertedManaCost','manaCost','loyalty','power','toughness',
    'type','types','subtypes','text',
    'brawl','commander','duel','future','historic','legacy','modern',
    'oldschool','pauper','penny','pioneer','standard','vintage'
]

  cards_df = pd.read_csv(CARD_DATA_PATH)\
    .query('mtgArenaId.notnull()')\
    .assign(Names=lambda df: df.name + '-' + df.setCode)\
    .assign(Names=lambda df: df.Names.apply(lambda x: x.replace(' ', '_').replace('//', 'II')))\
    .fillna('0')\
    [merge_cols]

  # Sort, merge, save cards in EFS and Dynamo
  cards = event['cards']
  for card in cards:
    staged_card = embed_df[['Names', card]]\
        .merge(cards_df, how='left', on='Names')\
        .sort_values(by=card, ascending=False)\
        .head(50)\
        .rename(columns={card: 'similarity'})\
        .assign(similarity=lambda df: df.similarity.astype('str'))\
        .assign(id=lambda df: df.id.astype('str'))\
        .assign(mtgArenaId=lambda df: df.mtgArenaId.astype('str'))\
        .assign(loyalty=lambda df: df.loyalty.astype('str'))\
        .assign(power=lambda df: df.power.astype('str'))\
        .assign(toughness=lambda df: df.toughness.astype('str'))\
        .assign(convertedManaCost=lambda df: df.convertedManaCost.astype('str'))
    
    # Write to EFS
    staged_card.to_csv(SORTED_CARD_PATH + '/{}.csv'.format(card), index=False)

    # Write to Dyanmo
    staged_card_dict = staged_card.to_dict(orient='records')
    Item = staged_card_dict[0]
    #Item['similarities'] = json.dumps(staged_card_dict[1:])
    Item['similarities'] = staged_card_dict[1:]

    _ = dynamodb_lib.call(SIMILARITY_TABLE, 'put_item', Item)
    sleep(1)

  return success({'status': True})