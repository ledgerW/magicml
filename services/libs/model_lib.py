from math import ceil
from pathlib import Path
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from .IncV3 import incV3
from .Vanilla import vanilla

model_opts = {
  'Vanilla': vanilla,
  'IncV3': incV3
}


def triplet_image_ds_from_dir(train_dir, batch_size, val_split, seed):
    train_df = pd.DataFrame({'path_obj': list(Path(train_dir).glob('*/*'))})\
        .assign(path_str=lambda df: df.path_obj.apply(str))

    if os.environ.get('SM_CURRENT_HOST'):
      train_df = train_df.assign(path_cls=lambda df: df.path_str.apply(lambda x: x.split('/')[-2]))
    else:
      train_df = train_df.assign(path_cls=lambda df: df.path_str.apply(lambda x: x.split('\\')[-2]))

    n_batches = ceil(train_df.shape[0] / batch_size)

    print('# of examples: {}'.format(train_df.shape[0]))
    print('# of classes: {}'.format(train_df.path_cls.nunique()))
    print('batch_size: {}'.format(batch_size))
    print('n_batches: {}'.format(n_batches))

    all_batches = []
    for batch_i in range(n_batches):
        if train_df.shape[0] <= batch_size:
            all_batches = all_batches + list(train_df.path_str)

            batched = train_df.path_str.values
            train_df = train_df.query('path_str not in @batched')
        else:
            n_samp_cls = min(int(batch_size/2), train_df.path_cls.nunique())
            samp_cls = np.random.choice(train_df.path_cls.unique(), n_samp_cls, replace=False)

            sample_df = train_df.query('path_cls in @samp_cls')

            batch_df = pd.DataFrame()
            for _ in range(2):
                half_batch = sample_df\
                    .sample(frac=1.0)\
                    .groupby('path_cls', as_index=False)\
                    .first()

                selected = half_batch.path_str.values

                sample_df = sample_df.query('path_str not in @selected')

                batch_df = pd.concat([batch_df, half_batch], axis=0)

            all_batches = all_batches + list(batch_df.path_str)

            batched = batch_df.path_str.values
            train_df = train_df.query('path_str not in @batched')

    # Get TF Dataset as Image-Label pairs
    train_ds = tf.data.Dataset.from_tensor_slices(all_batches)
    train_ds = train_ds.map(get_img_label_pair)

    #AUTOTUNE = tf.data.experimental.AUTOTUNE
    if val_split:
      # batch
      train_ds = train_ds.batch(batch_size, drop_remainder=True)

      # train-val split
      n_val_batches = ceil(n_batches*val_split)
      train_ds = train_ds.skip(n_val_batches)
      val_ds = train_ds.take(n_val_batches)

      # configure performance
      #train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
      #val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

      print('# of training batches: {}'.format(len(train_ds)))
      print('# of validation batches: {}'.format(len(val_ds)))

      return train_ds, val_ds
    else:
      # batch
      train_ds = train_ds.batch(batch_size)

      # configure performance
      #train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

      print('# of test batches: {}'.format(len(train_ds)))

      return train_ds


def get_img_label_pair(file_path):
  if os.environ.get('SM_CURRENT_HOST'):
    label = tf.strings.split(file_path, '/')[-2]
  else:
    label = tf.strings.split(file_path, '\\')[-2]

  img = tf.io.read_file(file_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.cast(img, tf.float32)
  
  return img, label


def build_preprocess_layer(input_sz, model):
  '''
  input_sz: raw image size to model (600x800)
  '''
  preprocess_layer = tf.keras.Sequential(name='preprocess')

  if model == 'IncV3':
    preprocess_layer.add(tf.keras.layers.Lambda(lambda x: tf.keras.applications.inception_v3.preprocess_input(x)))
  else:
    preprocess_layer.add(tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x, name=None)))
    preprocess_layer.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255))
  
  preprocess_layer.add(tf.keras.layers.experimental.preprocessing.Resizing(int(input_sz), int(input_sz)))

  return preprocess_layer


def build_augmentation_layer(input_sz):
  '''
  input_sz: image size to the network (i.e. post-preprocess)
  '''
  # Resize to +20% of network input size THEN IF training: RandomCrop to 256 ELSE Resize to 256
  augmentation_layer = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(input_sz[0], input_sz[1], seed=None, name=None)
  ], name='augmentation')

  return augmentation_layer


def build_model(flavor=None, input_size=256, embed_size=128, n_workers=1):
    model = model_opts[flavor](input_size=input_size, embed_size=embed_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001 * n_workers),
        loss=tfa.losses.TripletSemiHardLoss()
    )

    return model