from collections import namedtuple

ModelParams = namedtuple('ModelParams', [
    'name',
    'num_classes',
])

default_model_params = ModelParams(**{
    'name': 'initial',
    'num_classes': 22,
})

TrainingParams = namedtuple('TrainingParams', [
    'lr',
    'batch_size',
    'num_batches',
])

default_training_params = TrainingParams(**{
    # First 10k epochs: 0.00003, second 10k: 0.000003
    'lr': 0.000003,
    'batch_size': 8,
    'num_batches': 10_000,
})
