import sys
sys.path.append('../..')
sys.setrecursionlimit(100000)

import os
import boto3
import pandas as pd
import json
import requests
from time import sleep
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import libs.dynamodb_lib as dynamodb_lib

CARDS_URL = os.environ['CARDS_URL']
CARDS_PER_PAGE = int(os.environ['CARDS_PER_PAGE'])
PAGES_PER_WORKER = int(os.environ['PAGES_PER_WORKER'])


def scryfall_to_dynamo(table, res):
    BATCH_LIMIT = int(os.environ['DYNAMO_BATCH_LIMIT'])
    dynamo_batches = [(first, first+BATCH_LIMIT) for first in list(range(0, CARDS_PER_PAGE, BATCH_LIMIT))]

    for batch in dynamo_batches:
        sleep(0.3)
        cards = res.json()['data'][batch[0]:batch[1]]
        for idx, card in enumerate(cards):
            for key, val in card.items():
                cards[idx][key] = dynamodb_lib.batch_format(val)
            cards[idx]['cardId'] = cards[idx]['id']

        # Save to Dynamo
        RequestItems = {
            table: [
                {
                    'PutRequest': {
                        'Item': card
                    }
                } for card in cards
            ]
        }

        try:
            dynamo_res = dynamodb_lib.call(table, 'batch_write_item', RequestItems)
            if len(dynamo_res['UnprocessedItems'])>0:
                logger.info('Number of UnprocessedItems: {}'.format(len(dynamo_res['UnprocessedItems'])))
        except:
            e = sys.exc_info()[0]
            logger.info(page)
            logger.info(e)

    return res.json()['has_more']


def scryfall_to_s3(bucket, start_page, prices_df):
    file_name = 'prices{}.json'.format(start_page)
    df_loc = '/tmp/{}'.format(file_name)
    prices_df.to_json(df_loc, orient='records')

    today = pd.Timestamp.today()
    year = today.year
    month = today.month
    day = today.day
    today = '{}-{}-{}'.format(year, month, day)

    s3_file_name = '{}/{}'.format(today, file_name)
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(df_loc, bucket, s3_file_name)
        status = True
    except:
        e = sys.exc_info()
        logging.info(e)
        status = False
    return status

'''
def scryfall_to_elastic(domain, res):
    if True:
        logger.info('Number of Records: {}'.format(len(event['Records'])))
        return success({'status': True})
    else:
        host = os.environ['CARD_SEARCH_DOMAIN_ENDPOINT']
        region = 'us-east-1'
        service = 'es'
        cred = boto3.Session().get_credentials()
        awsauth = AWS4Auth(cred.access_key, cred.secret_key, region, service)

        es = Elasticsearch(hosts = [{'host': host, 'port': 443}],
                           http_auth = awsauth,
                           use_ssl = True,
                           verify_certs = True,
                           connection_class = RequestsHttpConnection)

        # COMPLETE ELASTIC SEARCH SYNC PROCESS!!!!
        for record in event['Records']:
            # Get the primary key for use as the Elasticsearch ID
            id = record['dynamodb']['Keys']['id']['S']

            if record['eventName'] == 'REMOVE':
                r = requests.delete(url + id, auth=awsauth)
            else:
                document = record['dynamodb']['NewImage']
                r = requests.put(url + id, auth=awsauth, json=document, headers=headers)
'''
