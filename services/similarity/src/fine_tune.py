import sys
sys.path.append('..')
sys.setrecursionlimit(100000)

import os
import pathlib
import tarfile
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel


MNT_PATH = '/opt/ml/processing'
LOCAL_INPUT_PATH = '{}/input'.format(MNT_PATH)
LOCAL_OUTPUT_PATH = '{}/output'.format(MNT_PATH)

# Fine Tuning Hyperparams
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
TB_DIR = LOCAL_OUTPUT_PATH + '/tensorboard'
EMBEDDING_SIZE = 64
MAX_INPUT_LENGTH = 125


def load_fine_tuning_data(target_dir, label_idx):
  texts = []
  labels = []
  for label_dir in label_idx.keys():
      for txt_file in (target_dir/label_dir).iterdir():
          texts.append(txt_file.read_text())
          labels.append(label_idx[label_dir])

  return texts, labels


def build_model(input_size, embedding_size, n_labels):
  bert = TFAutoModel.from_pretrained("bert-base-cased", output_hidden_states=False)

  input_ids = tf.keras.layers.Input(shape=(input_size,), name='input_ids', dtype='int32')
  input_token_types = tf.keras.layers.Input(shape=(input_size,), name='token_type_ids', dtype='int32')
  input_masks = tf.keras.layers.Input(shape=(input_size,), name='attention_mask', dtype='int32')

  x = bert.bert(input_ids, input_token_types, input_masks)[0]
  x = tf.keras.layers.GlobalMaxPool1D()(x)
  x = tf.keras.layers.Dense(embedding_size, activation='relu', name='embeddings')(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(n_labels)(x)

  clf_model = tf.keras.Model(
    inputs=[input_ids, input_token_types, input_masks],
    outputs = x
  )
  print(clf_model.summary())

  return clf_model


if __name__=="__main__":
  merge_cols = [
    'Names','id','mtgArenaId','scryfallId','name','colors','setName',
    'convertedManaCost','manaCost','loyalty','power','toughness',
    'type','types','subtypes','text','image_urls',
    'brawl','commander','duel','future','historic','legacy','modern',
    'oldschool','pauper','penny','pioneer','standard','vintage'
  ]

  cards_df = pd.read_csv(LOCAL_INPUT_PATH + '/cards.csv')\
    .assign(Names=lambda df: df.name + '-' + df.id.astype('str'))\
    .assign(Names=lambda df: df.Names.apply(lambda x: x.replace(' ', '_').replace('//', 'II')))\
    .fillna('0')\
    [merge_cols]

  train_cols = ['colors','types','text']

  train_df = cards_df[train_cols]\
    .assign(cls=lambda df: (df.colors + '-' + df.types).str.replace(',','_'))\
    [['cls','text']]

  print('Unique cls: {}'.format(train_df.cls.nunique()))

  # Organize cards into training data class directories
  cls_list = list(train_df.cls.unique())

  base_dir = 'fine_tune'
  target_dir = 'cls'
  cls_dir = pathlib.Path(LOCAL_INPUT_PATH, base_dir, target_dir)

  # make class directories
  for c in train_df.cls.unique():
      (cls_dir/c).mkdir(parents=True, exist_ok=True)

  # make txt file for each card
  for i, z in enumerate(zip(train_df.cls, train_df.text)):
    c, txt = z
    f = '{}_{}.txt'.format(i, c)
    (cls_dir/c/f).write_text(txt, encoding='utf-8')

  # load training data (no val or test since there will be no unseen data)
  cls_list = list(train_df.cls.unique())
  cls_idx = {cls_list[idx]: idx for idx in range(len(cls_list))}

  train_texts, train_labels = load_fine_tuning_data(cls_dir, cls_idx)

  # Prep Model
  tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

  train_encodings = tokenizer(train_texts, padding='max_length', max_length=MAX_INPUT_LENGTH)

  train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
  ))

  #input_size = train_dataset.element_spec[0]['input_ids'].shape[0]
  #print('Input Size (max token sequence length): {}'.format(input_size))

  n_labels = len(set(train_labels))
  print('N Train Labels: {}'.format(n_labels))

  # load huggingface pretrained BERT 
  clf_model = build_model(MAX_INPUT_LENGTH, EMBEDDING_SIZE, n_labels)

  clf_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)
    ],
  )

  # fine tune
  history = clf_model.fit(
    train_dataset.shuffle(1000).batch(BATCH_SIZE),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir=TB_DIR)]
  )

  # create embedding model
  model = tf.keras.Model(
    inputs=clf_model.inputs,
    outputs = clf_model.get_layer('embeddings').output
  )

  # save as TF saved_model and tar it up
  saved_model_path = 'MTG_BERT/1'
  model.save(saved_model_path)
  with tarfile.open(LOCAL_OUTPUT_PATH + '/model.tar.gz', "w:gz") as tar:
    tar.add(saved_model_path, arcname=os.path.basename(saved_model_path))

  # save tokenizer as well for free text query
  tokenizer_path = 'tokenizer'
  tokenizer.save_pretrained(tokenizer_path)
  with tarfile.open(LOCAL_OUTPUT_PATH + '/tokenizer.tar.gz', "w:gz") as tar:
    tar.add(tokenizer_path, arcname=os.path.basename(tokenizer_path))

  #=====================
  # Get embeddings for all cards with fine-tuned model
  cards_df = pd.read_csv(LOCAL_INPUT_PATH + '/cards.csv')\
    .reset_index(drop=True)\
    .fillna(value={'text': 'Blank'})

  cards_txt = list(cards_df.text)
  cards_name = [
    (name + '-' + str(id_val)).replace(' ','_').replace('//', 'II') for name, id_val in zip(cards_df.name, cards_df.id)
  ]

  card_txt_tokens = tokenizer(cards_txt, padding='max_length', max_length=MAX_INPUT_LENGTH, return_tensors="tf")
  card_txt_tokens = dict(card_txt_tokens)

  embeddings = model.predict(card_txt_tokens, batch_size=BATCH_SIZE)
  np.save(LOCAL_OUTPUT_PATH + '/embeddings.npy', embeddings)

  corr = np.inner(embeddings, embeddings)

  pd.DataFrame(corr, columns=cards_name, index=cards_name)\
    .to_parquet(LOCAL_OUTPUT_PATH + '/cards_embeddings.parquet')
