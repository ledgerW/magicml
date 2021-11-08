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


def fine_tune(event, context):
  '''
  '''
  MNT_PATH = '/opt/ml/processing'
  LOCAL_INPUT_PATH = '{}/input'.format(MNT_PATH)
  LOCAL_CODE_PATH = '{}/input/code'.format(MNT_PATH)
  LOCAL_OUTPUT_PATH = '{}/output'.format(MNT_PATH)

  image_uri = '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-training:2.5.1-transformers4.12.3-gpu-py37-cu112-ubuntu18.04'

  input_prefix = 'cards'
  input_data = 's3://{}/{}/cards.csv'.format(CLEAN_BUCKET, input_prefix)

  src_prefix = 'sm_processing'
  src_code = 's3://{}/{}/fine_tune.py'.format(SRC_BUCKET, src_prefix)

  output_prefix = 'MTG_BERT'
  output_data = 's3://{}/{}'.format(INFERENCE_BUCKET, output_prefix)

  s3.upload_file('src/fine_tune.py', SRC_BUCKET, 'sm_processing/fine_tune.py')

  now = datetime.datetime.now().strftime(format='%Y-%d-%m-%H-%M-%S')

  sm.create_processing_job(
    ProcessingJobName='fine-tune-MTG-BERT-{}'.format(now),
    RoleArn=SM_ROLE,
    StoppingCondition={
        'MaxRuntimeInSeconds': 7200
    },
    AppSpecification={
        'ImageUri': image_uri,
        'ContainerEntrypoint': [
            'python3',
            '-v',
            (LOCAL_CODE_PATH + '/fine_tune.py')
        ]
    },
    ProcessingResources={
        'ClusterConfig': {
            'InstanceCount': 1,
            'InstanceType': 'ml.g4dn.4xlarge',
            'VolumeSizeInGB': 30
        }
    },
    ProcessingInputs=[
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
  LOCAL_MODEL_PATH = '{}/models/MTG_BERT'.format(MNT_PATH)
  MODEL_TAR_PATH = LOCAL_MODEL_PATH + '/model.tar.gz'
  MODEL_PATH = LOCAL_MODEL_PATH + '/1'
  TOKENIZER_TAR_PATH = LOCAL_MODEL_PATH + '/tokenizer.tar.gz'
  TOKENIZER_PATH = LOCAL_MODEL_PATH + '/tokenizer'
  CORR_MATRIX_PATH = LOCAL_INPUT_PATH + '/corr_matrix.parquet'
  EMBEDDINGS_PATH = LOCAL_INPUT_PATH + '/embeddings.npy'
  CARD_DATA_PATH = LOCAL_INPUT_PATH + '/cards.csv'
  SORTED_CARD_PATH = LOCAL_OUTPUT_PATH + '/sorted'

  os.makedirs(LOCAL_INPUT_PATH, exist_ok=True)
  os.makedirs(LOCAL_OUTPUT_PATH, exist_ok=True)
  os.makedirs(SORTED_CARD_PATH, exist_ok=True)
  os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
  os.makedirs(MODEL_PATH, exist_ok=True)

  # Get embeddings and card data from S3
  #download_tasks = [
  #  [MODELS_BUCKET, 'use-large/model.tar.gz', USE_TAR_PATH],
  #  [INFERENCE_BUCKET, 'use-large/cards_embeddings.csv', CORR_MATRIX_PATH],
  #  [INFERENCE_BUCKET, 'use-large/embeddings.npy', EMBEDDINGS_PATH],
  #  [CLEAN_BUCKET, 'cards/cards.csv', CARD_DATA_PATH]
  #]
  #for task in download_tasks:
  #  print('task')
  #  payload = {'download': {
  #    'bucket': task[0],
  #    'key': task[1],
  #    'path': task[2]
  #  }}

  #  res = lambda_client.invoke(
  #      FunctionName='magicml-similarity-{}-stage_embed_worker'.format(STAGE),
  #      InvocationType='Event',
  #      Payload=json.dumps(payload)
  #  )
  #  sleep(0.2)

  # check if S3 Triggered
  if not event.get('n_cards'):
    event['n_cards'] = -1
    event['batch_size'] = 100

  print('model')
  s3.download_file(INFERENCE_BUCKET, 'MTG_BERT/model.tar.gz', MODEL_TAR_PATH)
  print('tokenizer')
  s3.download_file(INFERENCE_BUCKET, 'MTG_BERT/tokenizer.tar.gz', TOKENIZER_TAR_PATH)
  print('embeddings.parquet')
  s3.download_file(INFERENCE_BUCKET, 'MTG_BERT/cards_embeddings.parquet', CORR_MATRIX_PATH)
  print('embeddings.npy')
  s3.download_file(INFERENCE_BUCKET, 'MTG_BERT/embeddings.npy', EMBEDDINGS_PATH)
  print('cards.csv')
  s3.download_file(CLEAN_BUCKET, 'cards/cards.csv', CARD_DATA_PATH)

  # untar model (for free_text_query api)
  os.system('tar -xf {} -C {}'.format(MODEL_TAR_PATH, MODEL_PATH))
  os.system('rm -r {}'.format(MODEL_TAR_PATH))

  # untar tokenizer (for free_text_query api)
  os.system('tar -xf {} -C {}'.format(TOKENIZER_TAR_PATH, TOKENIZER_PATH))
  os.system('rm -r {}'.format(TOKENIZER_TAR_PATH))

  # Get card embeddings matrix
  all_cards = pd.read_parquet(CORR_MATRIX_PATH)\
    .reset_index()\
    .rename(columns={'index': 'Names'})\
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
    sleep(0.2)

  return success({'status': True})


def stage_embed_worker(event, context):
  '''
  event['cards']: list of card names to handle
  '''
  MNT_PATH = os.getenv('EFS_MOUNT_PATH')
  LOCAL_INPUT_PATH = '{}/input'.format(MNT_PATH)
  LOCAL_OUTPUT_PATH = '{}/output'.format(MNT_PATH)
  LOCAL_MODEL_PATH = '{}/models/MTG_BERT'.format(MNT_PATH)
  MODEL_TAR_PATH = LOCAL_MODEL_PATH + '/model.tar.gz'
  MODEL_PATH = LOCAL_MODEL_PATH + '/1'
  CORR_MATRIX_PATH = LOCAL_INPUT_PATH + '/corr_matrix.parquet'
  EMBEDDINGS_PATH = LOCAL_INPUT_PATH + '/embeddings.npy'
  CARD_DATA_PATH = LOCAL_INPUT_PATH + '/cards.csv'
  SORTED_CARD_PATH = LOCAL_OUTPUT_PATH + '/sorted'

  # handle download from S3 task
  if event.get('download'):
    print('download task')
    bucket = event['download']['bucket']
    key = event['download']['key']
    path = event['download']['path']
    s3.download_file(bucket, key, path)

    if 'model.tar.gz' in key:
      # untar model (for free_text_query api)
      os.system('tar -xf {} -C {}'.format(MODEL_TAR_PATH, MODEL_PATH))
      os.system('rm -r {}'.format(MODEL_TAR_PATH))
  else:
    print('batch of cards task')
    # Get card embeddings matrix
    embed_df = pd.read_parquet(CORR_MATRIX_PATH)\
      .reset_index()\
      .rename(columns={'index': 'Names'})\

    # Get MTGJSON clean cards data
    merge_cols = [
      'Names','id','mtgArenaId','scryfallId','name','colors','setName',
      'convertedManaCost','manaCost','loyalty','power','toughness',
      'type','types','subtypes','text','image_urls',
      'brawl','commander','duel','future','historic','legacy','modern',
      'oldschool','pauper','penny','pioneer','standard','vintage'
    ]

    cards_df = pd.read_csv(CARD_DATA_PATH)\
      .assign(Names=lambda df: df.name + '-' + df.id.astype('str'))\
      .assign(Names=lambda df: df.Names.apply(lambda x: x.replace(' ', '_').replace('//', 'II')))\
      .fillna('0')\
      [merge_cols]

    # Sort, merge, save cards in EFS and Dynamo
    cards = event['cards']
    for card in cards:
      print(card)
      staged_card = embed_df[['Names', card]]\
        .merge(cards_df, how='left', on='Names')

      # Item (this card) to be stored in Dynamo
      Item = staged_card.query('Names == @card')\
        .rename(columns={card: 'similarity'})\
        .assign(similarity=lambda df: df.similarity.astype('str'))\
        .assign(id=lambda df: df.id.astype('str'))\
        .assign(mtgArenaId=lambda df: df.mtgArenaId.astype('str'))\
        .assign(loyalty=lambda df: df.loyalty.astype('str'))\
        .assign(power=lambda df: df.power.astype('str'))\
        .assign(toughness=lambda df: df.toughness.astype('str'))\
        .assign(convertedManaCost=lambda df: df.convertedManaCost.astype('str'))\
        .to_dict(orient='records')[0]

      staged_card = staged_card\
        .sort_values(by=card, ascending=False)\
        .head(51)\
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

      # Write to Dyanmo (exclude this card from it's own similarities)
      Item['similarities'] = staged_card.query('Names != @card').to_dict(orient='records')

      _ = dynamodb_lib.call(SIMILARITY_TABLE, 'put_item', Item)
      sleep(1)

  return success({'status': True})