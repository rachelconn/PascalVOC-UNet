from data_loader import load_dataset
from model import Model
from config import default_model_params, default_training_params

if __name__ == '__main__':
    train_ds, validation_ds, test_ds = load_dataset()
    model = Model(default_model_params, default_training_params)
    model.train(train_ds, validation_ds)
