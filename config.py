from collections import namedtuple

ModelParams = namedtuple('ModelParams', [
    'name',
    'num_classes',
])

default_model_params = ModelParams(**{
    'name': 'resnet',
    'num_classes': 21,
})

TrainingParams = namedtuple('TrainingParams', [
    'lr',
    'batch_size',
    'num_epochs',
    'weight_decay',
])

default_training_params = TrainingParams(**{
    'lr': 3e-6,
    'batch_size': 8,
    'num_epochs': 400,
    'weight_decay': 1e-4,
})
