
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_data_loader(dataset_name: str, batch_size: int = 32):
    if dataset_name == 'cifar-100':
        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        # 加载CIFAR - 100数据集
        full_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform)
        train_size = int(0.8 * len(full_set))
        val_size = len(full_set) - train_size
        train_set, val_set = random_split(full_set, [train_size, val_size])

        test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader