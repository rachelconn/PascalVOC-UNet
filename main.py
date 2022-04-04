import os
from data_loader import load_pascal_dataset, load_outcrop_dataset
from model import Model
from config import default_model_params, default_training_params

OUTCROP_DATASET_FOLDER = os.path.join('E:', 'outcrop')

if __name__ == '__main__':
    train_ds, validation_ds, test_ds = load_pascal_dataset()
    model = Model(default_model_params, default_training_params)

    model.train(train_ds, validation_ds)

    model.test(test_ds)

    # TODO: swap to this code instead when we feel comfortable using the outcrop dataset
    # train_ds, validation_ds, test_ds = load_outcrop_dataset(OUTCROP_DATASET_FOLDER)
    # for x, y in train_ds:
    #     print(x.shape, y.shape)
