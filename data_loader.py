import os
import tensorflow as tf

TRAIN_VAL_DATASET_PATH = os.path.abspath(os.path.join('datasets', 'VOC2012'))
LABEL_DIRECTORY = os.path.join(TRAIN_VAL_DATASET_PATH, 'SegmentationClass')

def _get_x_y(label_path):
    image_path = tf.strings.regex_replace(label_path, 'SegmentationClass', 'JPEGImages')
    image_path = tf.strings.regex_replace(image_path, '.png', '.jpg')
    return image_path, label_path

def load_dataset(path=TRAIN_VAL_DATASET_PATH):
    file_ds = tf.data.Dataset.list_files(os.path.join(LABEL_DIRECTORY, '*'), shuffle=False)
    # TODO: take (256, 256 subsections)
    dataset = file_ds.map(_get_x_y)
    return dataset
