"""
数据加载器工具
提供统一的数据加载接口
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Optional


def get_mnist_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    获取 MNIST 数据加载器

    Args:
        data_dir: 数据集目录
        batch_size: 批次大小
        num_workers: 数据加载进程数（CPU 建议设为 0）
        shuffle_train: 是否打乱训练集

    Returns:
        train_loader, test_loader
    """
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
    ])

    # 加载数据集
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
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
    print("测试 MNIST 数据加载器...")
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    # 获取一个批次
    images, labels = next(iter(train_loader))
    print(f"[SUCCESS] MNIST 加载成功:")
    print(f"   批次形状: {images.shape}")
    print(f"   标签形状: {labels.shape}")
    print(f"   训练批次数: {len(train_loader)}")
    print(f"   测试批次数: {len(test_loader)}")

    # 打印数据集信息
    print("\n数据集信息:")
    for dataset in ['mnist', 'cifar10', 'squad']:
        info = get_dataset_info(dataset)
        if info:
            print(f"\n{info['name']}:")
            print(f"  类别数: {info['num_classes']}")
            print(f"  输入形状: {info['input_shape']}")
            print(f"  训练集大小: {info['train_size']}")
            print(f"  硬件需求: {info['hardware']}")
