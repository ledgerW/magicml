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


def query(event, context):
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


