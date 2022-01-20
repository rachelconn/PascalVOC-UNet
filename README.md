# Setup
This was tested using 64-bit Python 3.8.5. Create a virtual environment if you want and run `pip install -r requirements.txt` in the root folder to install dependencies.

Download the training/validation data from [this link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit),
create a directory called `datasets` in the root of this project, and extract the `VOC2012` folder to the datasets folder (so its subfolders are located in `datasets/VOC2012` — for instance, `datasets/VOC2012/JPEGImages`).

# Running
You can configure hyperparameters and other variables by modifying `default_model_params` and `default_training_params` in `config.py`.
Then, run `python main.py` to train and evaluate your model.

# Structure
`main.py`: Trains and evaluates a model.
`config.py`: Sets hyperparameters and other config data.
`model.py`: Contains the code to setup, train, and test models on PASCAL VOC 2012 data.
`network.py`: Contains the UNet architecture used to make predictions.
`data_loader.py`: Creates split datasets for training/validation/testing on PASCAL VOC 2012 data.
