from collections import namedtuple

ModelParams = namedtuple('ModelParams', [
    'name',
    'num_classes',
])

default_model_params = ModelParams(**{
    'name': 'deep_encoder',
    'num_classes': 22,
})

TrainingParams = namedtuple('TrainingParams', [
    'lr',
    'batch_size',
    'num_batches',
    'weight_decay',
])

default_training_params = TrainingParams(**{
    # First 10k epochs: 0.00003, second 10k: 0.000003
    'lr': 3e-4,
    'batch_size': 4,
    'num_batches': 80_000,
    'weight_decay': 1e-4,
})
