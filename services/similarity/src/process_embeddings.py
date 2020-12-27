import sys
sys.path.append('..')
sys.setrecursionlimit(100000)

import os
import json
import pathlib
import tarfile
import boto3
import numpy as np
import pandas as pd
import tensorflow as tf


MNT_PATH = '/opt/ml/processing'
LOCAL_INPUT_PATH = '{}/input'.format(MNT_PATH)
LOCAL_MODEL_PATH = '{}/model'.format(MNT_PATH)
LOCAL_OUTPUT_PATH = '{}/output'.format(MNT_PATH)


if __name__=="__main__":
    # Get all embeddings
    cards_df = pd.read_csv(LOCAL_INPUT_PATH + '/cards.csv')

    arena_df = cards_df.query('mtgArenaId.notnull()')\
      .reset_index(drop=True)\
      .fillna(value={'text': 'Blank'})

    arena_txt = list(arena_df.text)
    arena_name = [
      (name + '-' + str(id_val)).replace(' ','_').replace('//', 'II') for name, id_val in zip(arena_df.name, arena_df.id)
    ]

    model_path = LOCAL_MODEL_PATH + '/model.tar.gz'
    print('Extracting model from path: {}'.format(model_path))
    with tarfile.open(model_path) as tar:
        tar.extractall(path='.')

    print('Loading model')
    use_embed = tf.saved_model.load('1')

    embeddings = use_embed(arena_txt)

    corr = np.inner(embeddings, embeddings)

    pd.DataFrame(corr, columns=arena_name, index=arena_name)\
      .to_csv(LOCAL_OUTPUT_PATH + '/arena_embeddings.csv')
