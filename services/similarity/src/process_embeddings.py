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
  cards_df = pd.read_csv(LOCAL_INPUT_PATH + '/cards.csv')\
    .reset_index(drop=True)\
    .fillna(value={'text': 'Blank'})

  cards_txt = list(cards_df.text)
  cards_name = [
    (name + '-' + str(id_val)).replace(' ','_').replace('//', 'II') for name, id_val in zip(cards_df.name, cards_df.id)
  ]

  model_path = LOCAL_MODEL_PATH + '/model.tar.gz'
  print('Extracting model from path: {}'.format(model_path))
  with tarfile.open(model_path) as tar:
      tar.extractall(path='.')

  print('Loading model')
  use_embed = tf.saved_model.load('1')

  embeddings = use_embed(cards_txt)
  np.save(LOCAL_OUTPUT_PATH + '/embeddings.npy', embeddings)

  corr = np.inner(embeddings, embeddings)

  pd.DataFrame(corr, columns=cards_name, index=cards_name)\
    .to_parquet(LOCAL_OUTPUT_PATH + '/cards_embeddings.parquet')
