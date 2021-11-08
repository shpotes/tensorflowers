# -*- coding: utf-8 -*-
"""KNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15k339aljUedc0_v6vcQ5yG9y4tkrPeRi
"""

! pip install datasets

from typing import Dict, Tuple
from datasets import load_dataset
import tensorflow as tf
import pandas as pd

def to_tf_data(dataset):
    dataset.set_format('tf')

    features = {
        x: dataset[x] for x in ['lat', 'lon', 'image']
    }

    return tf.data.Dataset.from_tensor_slices((features, dataset['labels']))

@tf.function
def load_image(row: tf.data.Dataset, labels: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], tf.Tensor]:
    img_path = row['image']
    img_buf = tf.io.read_file(img_path)
    img = tf.io.decode_png(img_buf)

    metadata = {
        'lat': row['lat'],
        'lon': row['lon']
    }

    return img, metadata, labels

@tf.function
def preprocessing(
    img: tf.data.Dataset, 
    _, 
    labels: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:

    preprocessed_img = img / 255
    one_hot_label = tf.one_hot(labels, depth=20)

    return preprocessed_img, one_hot_label

def get_raw_dataset(with_preprocessing: bool = True) -> tf.data.Dataset:
    ds = load_dataset("shpotes/tfcol")

    train_ds = to_tf_data(ds['train'])
    val_ds = to_tf_data(ds['validation'])

    train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if with_preprocessing:
        train_ds = train_ds.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, val_ds

train_ds, validation_ds = get_raw_dataset(False)

lats = []
longs = []
labels = []

for i in train_ds.as_numpy_iterator():
    lats.append(i[1]['lat'])
    longs.append(i[1]['lon'])
    labels.append(i[2])

spatial_df = pd.DataFrame({'lat': lats, 'lon': longs, 'label': labels})
spatial_df = spatial_df.explode('label')
spatial_df.head()

X_train = spatial_df.drop('label', axis=1)
y_train = spatial_df['label']

lats = []
longs = []
labels = []

for i in validation_ds.as_numpy_iterator():
    lats.append(i[1]['lat'])
    longs.append(i[1]['lon'])
    labels.append(i[2])

label_df = pd.DataFrame({'lat': lats, 'lon': longs, 'label': labels})
label_df = label_df.explode('label')
label_df.head()

X_test = label_df.drop('label', axis=1)
y_test = label_df['label']

len(X_train), len(y_train), len(X_test), len(y_test)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 6)

y_train = y_train.astype('int')
model.fit(X_train, y_train)

y_test = y_test.astype('int')
acc = model.score(X_test, y_test)
print(acc)

predicted = model.predict(X_test)

for i in range(len(predicted)):
    print("Predicted: ", predicted[i], "Real: ", y_test.iloc[i])

y_test.value_counts()

from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
cf = confusion_matrix(y_test, predicted)
cf = cf / cf.astype(np.float).sum(axis=1)
sns.heatmap(data = cf)

from sklearn.metrics import f1_score
f1_score(y_test, predicted, average=None)