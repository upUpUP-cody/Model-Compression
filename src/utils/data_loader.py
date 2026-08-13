"""
数据加载器工具
提供统一的数据加载接口
"""
import hashlib
import json
from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def get_mnist_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle_train: bool = True,
    validation_fraction: float = 0.1,
    split_seed: int = 42,
    return_test: bool = False,
) -> Tuple[DataLoader, DataLoader] | Tuple[DataLoader, DataLoader, DataLoader]:
    """
    获取 MNIST 数据加载器

    Args:
        data_dir: 数据集目录
        batch_size: 批次大小
        num_workers: 数据加载进程数（CPU 建议设为 0）
        shuffle_train: 是否打乱训练集

    Returns:
        train_loader, validation_loader, and optionally the official test_loader
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    total_size = len(full_train_dataset)
    validation_size = int(total_size * validation_fraction)
    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    train_dataset = Subset(full_train_dataset, train_indices)
    validation_dataset = Subset(full_train_dataset, validation_indices)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=shuffle_train, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    if return_test:
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        return train_loader, validation_loader, test_loader
    return train_loader, validation_loader


def mnist_split_metadata(train_loader: DataLoader, validation_loader: DataLoader, split_seed: int) -> Dict[str, object]:
    """Return a stable hash and sizes for a train/validation split."""
    def indices(loader: DataLoader) -> list[int]:
        dataset = loader.dataset
        if not isinstance(dataset, Subset):
            raise ValueError("loader dataset must be a Subset")
        return [int(index) for index in dataset.indices]

    payload = {
        "split_seed": int(split_seed),
        "train_indices": indices(train_loader),
        "validation_indices": indices(validation_loader),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "split_seed": int(split_seed),
        "train_size": len(payload["train_indices"]),
        "validation_size": len(payload["validation_indices"]),
        "split_hash": hashlib.sha256(encoded).hexdigest(),
    }


def get_cifar10_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 0,
    augmentation: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    获取 CIFAR-10 数据加载器

    Args:
        data_dir: 数据集目录
        batch_size: 批次大小
        num_workers: 数据加载进程数
        augmentation: 是否使用数据增强

    Returns:
        train_loader, test_loader
    """
    # 训练集转换（带数据增强）
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])

    # 测试集转换（无数据增强）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    # 加载数据集
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader


def get_dataset_info(dataset_name: str) -> dict:
    """
    获取数据集基本信息

    Args:
        dataset_name: 数据集名称 ('mnist', 'cifar10', 'squad')

    Returns:
        包含数据集信息的字典
    """
    info = {
        'mnist': {
            'name': 'MNIST',
            'num_classes': 10,
            'input_shape': (1, 28, 28),
            'train_size': 60000,
            'test_size': 10000,
            'hardware': '[CPU OK]',
            'description': '手写数字识别'
        },
        'cifar10': {
            'name': 'CIFAR-10',
            'num_classes': 10,
            'input_shape': (3, 32, 32),
            'train_size': 50000,
            'test_size': 10000,
            'hardware': '[GPU Recommended]',
            'description': '小型图像分类'
        },
        'squad': {
            'name': 'SQuAD 2.0',
            'num_classes': None,
            'input_shape': None,
            'train_size': 130319,
            'test_size': 11873,
            'hardware': '[GPU Required]',
            'description': '阅读理解问答'
        }
    }

    return info.get(dataset_name.lower(), None)


if __name__ == '__main__':
    # 测试数据加载器
    print(" MNIST ...")
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    # 获取一个批次
    images, labels = next(iter(train_loader))
    print(f"[SUCCESS] MNIST :")
    print(f"   : {images.shape}")
    print(f"   : {labels.shape}")
    print(f"   : {len(train_loader)}")
    print(f"   : {len(test_loader)}")

    # 打印数据集信息
    print("\n:")
    for dataset in ['mnist', 'cifar10', 'squad']:
        info = get_dataset_info(dataset)
        if info:
            print(f"\n{info['name']}:")
            print(f"  : {info['num_classes']}")
            print(f"  : {info['input_shape']}")
            print(f"  : {info['train_size']}")
            print(f"  : {info['hardware']}")
