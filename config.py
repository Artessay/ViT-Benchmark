CONFIG = {
    'datasets': [
        {
            'name': 'imagenet',
            'input_size': 224,
            'num_classes': 1000,
            'model_mapping': 'vit_base_patch16_224'
        },
        {
            'name': 'cifar10',
            'input_size': 32,
            'num_classes': 10,
            'model_mapping': 'vit_small_patch16_224'
        },
        {
            'name': 'cifar100',
            'input_size': 32,
            'num_classes': 100,
            'model_mapping': 'vit_small_patch16_224'
        }
    ],
    'data_root': 'data/',
    'batch_size': 32,
    'epochs': 10,
    'output_file': 'results.csv'
}