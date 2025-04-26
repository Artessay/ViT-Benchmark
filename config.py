CONFIG = {
    'cifar-10': {
        'epochs': 50, 
        'patience': 5,
        'lr': 3e-5,
        'batch_size': 512,
        'num_classes': 10,
        'model_name': 'vit_b_16'
    },
    'cifar-100': {
        'epochs': 50, 
        'patience': 5,
        'lr': 3e-5,
        'batch_size': 512,
        'num_classes': 100,
        'model_name': 'vit_b_16'
    },
}