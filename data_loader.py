import os
import random
import numpy as np
import tensorflow as tf

TRAIN_VAL_DATASET_PATH = os.path.abspath(os.path.join('datasets', 'VOC2012'))
LABEL_DIRECTORY = os.path.join(TRAIN_VAL_DATASET_PATH, 'SegmentationClass')

def _get_x_y(label_path):
    image_path = tf.strings.regex_replace(label_path, 'SegmentationClass', 'JPEGImages')
    image_path = tf.strings.regex_replace(image_path, '.png', '.jpg')
    return image_path, label_path

def load_pascal_dataset(path=TRAIN_VAL_DATASET_PATH):
    # Load datset
    file_ds = tf.data.Dataset.list_files(os.path.join(LABEL_DIRECTORY, '*'), shuffle=False)
    dataset = file_ds.map(_get_x_y)

    # Split into training/validation/test
    ds_size = tf.data.experimental.cardinality(dataset).numpy()
    train_ratio = 0.9
    train_size = train_ratio * ds_size
    validation_ratio = 0.05
    validation_size = validation_ratio * ds_size
    test_ratio = 0.05
    train_ds = dataset.take(train_size).shuffle(100)
    validation_ds = dataset.skip(train_size).take(validation_size)
    test_ds = dataset.skip(train_size).skip(validation_size)

    return train_ds, validation_ds, test_ds

def parse_outcrop_label_file(filename):
    with open(filename) as f:
        resolution, _, mask = f.readlines()
        width, height = (int(x) for x in resolution.split(','))
        mask = np.array(mask.split(','), dtype=np.int16).reshape(width, height, 2)
        return mask

def generate_outcrop_dataset(path):
    x_dir = os.path.join(path, b'x')
    y_dir = os.path.join(path, b'y')
    x_basenames = os.listdir(x_dir)
    random.shuffle(x_basenames)
    for x_basename in x_basenames:
        x = tf.keras.utils.load_img(os.path.join(x_dir, x_basename))
        x = np.array(x) / 255

        y_filename = os.path.join(y_dir, os.path.splitext(x_basename)[0] + b'.csv')
        y = parse_outcrop_label_file(y_filename)

        yield x, y

def load_outcrop_dataset(path, train_split=0.8, val_split=0.1, test_split=0.1):
    # Calculate length to create splits
    dataset_size = len(os.listdir(os.path.join(path, 'x')))
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = int(test_split * dataset_size)

    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        generate_outcrop_dataset,
        args=(path,),
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 2), dtype=tf.int16),
        ),
    )

    # Split into train, val, test sets
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    test_dataset = val_dataset.skip(val_size)
    val_dataset = test_dataset.take(val_size)

    return train_dataset, val_dataset, test_dataset
