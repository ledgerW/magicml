import boto3
import os
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch

patch(['boto3'])

def call(table, action, params):
    # check if running local for dev/testing
    if os.getenv('IS_LOCAL'):
        dynamodb = boto3.resource('dynamodb', region_name='localhost', endpoint_url='http://localhost:8000')
        client = boto3.client('dynamodb', region_name='localhost', endpoint_url='http://localhost:8000')
    else:
        dynamodb = boto3.resource('dynamodb')
        client = boto3.client('dynamodb')

    table = dynamodb.Table(table)

    try:
        xray_recorder.begin_subsegment(action)
        if action == 'put_item':
            return getattr(table, action)(Item=params)
        elif action == 'get_item' or action == 'delete_item':
            return getattr(table, action)(Key=params)
        elif action == 'query':
            return getattr(table, action)(KeyConditionExpression=params['KeyConditionExpression'])
        elif action == 'update_item':
            return getattr(table, action)(Key=params['Key'],
                                          UpdateExpression=params['UpdateExpression'],
                                          ExpressionAttributeValues=params['ExpressionAttributeValues'],
                                          ReturnValues=params['ReturnValues'])
        elif action == 'batch_write_item':
            return getattr(client, action)(RequestItems=params)
    except Exception as e:
        e = sys.exc_info()[0]
        logger.info(page)
        logger.info(e)
        xray_recorder.end_subsegment()


def batch_format(val):
    if type(val) is str:
        return {"S": str(val)} if len(val)>0 else {"NULL": True}
    if type(val) is int:
        return {"N": str(val)}
    if type(val) is float:
        return {"N": str(val)}
    if type(val) is list:
        if len(val)>0:
            if type(val[0]) is str:
                return {"SS": val}
            if type(val[0]) is int:
                return {"NS": [str(item) for item in val]}
            if type(val[0]) is float:
                return {"NS": [str(item) for item in val]}
            if type(val[0]) is dict:
                return {
                        "M": {
                                str(idx): {
                                    "M": {dict_key: batch_format(dict_val) for dict_key, dict_val in dict_item.items()}
                                } for idx, dict_item in enumerate(val)
                            }
                        }
        else:
            return {"SS": ['empty']}
    if type(val) is bool:
        return {"BOOL": True}
    if val is None:
        return {"NULL": True}
    if type(val) is dict:
        return {"M": {val_key: batch_format(val_val) for val_key, val_val in val.items()}}
