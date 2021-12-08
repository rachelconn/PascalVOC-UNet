from data_loader import load_dataset
from model import Model
from config import default_model_params, default_training_params

if __name__ == '__main__':
    dataset = load_dataset()
    model = Model(default_model_params, default_training_params)
    model.train(dataset)
