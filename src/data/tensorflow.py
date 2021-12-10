from typing import Dict, Tuple
from datasets import load_dataset
import tensorflow as tf

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
