import os
from timm.data import create_dataset, create_loader, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def get_data_loader(dataset_name, batch_size=32, input_size=224):
    """
    获取指定数据集的数据加载器
    Args:
        dataset_name (str): 数据集名称 (imagenet, cifar10, cifar100)
        batch_size (int): 批处理大小
        input_size (int): 输入图像尺寸
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 数据集路径配置（需要根据实际路径修改）
    name = f"torch/{dataset_name}"
    data_dir = f"data/{dataset_name}"
    
    # 创建数据集变换
        # 添加分辨率适配器
    if input_size < 224:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            create_transform(
                input_size=224,
                is_training=True,
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD
            )
        ])
    else:
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        )

    # 创建训练集和验证集
    train_dataset = create_dataset(
        root=os.path.join(data_dir, 'train'),
        name=name,
        split='train',
        transform=transform,
        download=True,
    )
    
    val_dataset = create_dataset(
        root=os.path.join(data_dir, 'val'),
        name=name,
        split='validation',
        transform=transform,
        download=True,
    )

    # 创建数据加载器
    train_loader = create_loader(
        train_dataset,
        batch_size=batch_size,
        is_training=True,
        num_workers=4
    )

    val_loader = create_loader(
        val_dataset,
        batch_size=batch_size*2,
        is_training=False,
        num_workers=4
    )

    return train_loader, val_loader