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
      def is_within_directory(directory, target):
          
          abs_directory = os.path.abspath(directory)
          abs_target = os.path.abspath(target)
      
          prefix = os.path.commonprefix([abs_directory, abs_target])
          
          return prefix == abs_directory
      
      def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
      
          for member in tar.getmembers():
              member_path = os.path.join(path, member.name)
              if not is_within_directory(path, member_path):
                  raise Exception("Attempted Path Traversal in Tar File")
      
          tar.extractall(path, members, numeric_owner=numeric_owner) 
          
      
      safe_extract(tar, path=".")

  print('Loading model')
  use_embed = tf.saved_model.load('1')

  embeddings = use_embed(cards_txt)

  corr = np.inner(embeddings, embeddings)

  pd.DataFrame(corr, columns=cards_name, index=cards_name)\
    .to_csv(LOCAL_OUTPUT_PATH + '/cards_embeddings.csv')
