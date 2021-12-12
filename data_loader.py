import os
import tensorflow as tf

TRAIN_VAL_DATASET_PATH = os.path.abspath(os.path.join('datasets', 'VOC2012'))
LABEL_DIRECTORY = os.path.join(TRAIN_VAL_DATASET_PATH, 'SegmentationClass')

def _get_x_y(label_path):
    image_path = tf.strings.regex_replace(label_path, 'SegmentationClass', 'JPEGImages')
    image_path = tf.strings.regex_replace(image_path, '.png', '.jpg')
    return image_path, label_path

def load_dataset(path=TRAIN_VAL_DATASET_PATH):
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
