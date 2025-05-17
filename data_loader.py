import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def get_data_loader(dataset_name: str, batch_size: int = 32):
    if dataset_name == "cifar-10":
        # 数据预处理
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        # 加载CIFAR - 10数据集
        full_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        train_size = int(0.8 * len(full_set))
        val_size = len(full_set) - train_size
        train_set, val_set = random_split(full_set, [train_size, val_size])

        test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "cifar-100":
        # 数据预处理
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        # 加载CIFAR - 100数据集
        full_set = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        train_size = int(0.8 * len(full_set))
        val_size = len(full_set) - train_size
        train_set, val_set = random_split(full_set, [train_size, val_size])

        test_set = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "imagenet":
        # 数据预处理
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # 加载ImageNet数据集
        full_set = torchvision.datasets.ImageNet(root="./data/imagenet", split="train", transform=transform)

        train_size = int(0.8 * len(full_set))
        val_size = len(full_set) - train_size
        train_set, val_set = random_split(full_set, [train_size, val_size])

        test_set = torchvision.datasets.ImageNet(root="./data/imagenet", split="val", transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader
