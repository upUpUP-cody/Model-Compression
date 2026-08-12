"""
数据集准备脚本
自动下载和准备 MNIST, CIFAR-10, SQuAD 数据集
"""
import os
import argparse
from pathlib import Path
import torch
from torchvision import datasets, transforms


def prepare_mnist(data_dir: str = './data'):
    """
    准备 MNIST 数据集
    [OK] CPU 友好，自动下载
    """
    print("\n" + "="*50)
    print(" MNIST ")
    print("="*50)

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # 下载训练集
    print("\n...")
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    # 下载测试集
    print("\n...")
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    print(f"\n[SUCCESS] MNIST :")
    print(f"   : {len(train_dataset)}")
    print(f"   : {len(test_dataset)}")
    print(f"   : 28x28")
    print(f"   : 10")
    print(f"   : {data_path / 'MNIST'}")

    return train_dataset, test_dataset


def prepare_cifar10(data_dir: str = './data'):
    """
    准备 CIFAR-10 数据集
    [WARN]️ CPU 可训练但慢，建议 GPU
    """
    print("\n" + "="*50)
    print(" CIFAR-10 ")
    print("="*50)

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # 下载训练集
    print("\n...")
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    # 下载测试集
    print("\n...")
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    print(f"\n[SUCCESS] CIFAR-10 :")
    print(f"   : {len(train_dataset)}")
    print(f"   : {len(test_dataset)}")
    print(f"   : 32x32x3")
    print(f"   : 10")
    print(f"   : {data_path / 'cifar-10-batches-py'}")

    return train_dataset, test_dataset


def prepare_squad(data_dir: str = './data'):
    """
    准备 SQuAD 2.0 数据集
    🔥 必须 GPU，CPU 不可行

    需要先安装: pip install datasets
    """
    print("\n" + "="*50)
    print(" SQuAD 2.0 ")
    print("="*50)

    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR]  datasets ")
        print("   : pip install datasets")
        return None, None

    data_path = Path(data_dir) / 'squad'
    data_path.mkdir(parents=True, exist_ok=True)

    # 下载 SQuAD 2.0
    print("\n SQuAD 2.0 ...")
    dataset = load_dataset('squad_v2', cache_dir=str(data_path))

    train_dataset = dataset['train']
    dev_dataset = dataset['validation']

    print(f"\n[SUCCESS] SQuAD 2.0 :")
    print(f"   : {len(train_dataset)}")
    print(f"   : {len(dev_dataset)}")
    print(f"   :  (QA)")
    print(f"   : {data_path}")
    print(f"\n[WARNING] SQuAD  GPU")

    return train_dataset, dev_dataset


def main():
    parser = argparse.ArgumentParser(description='准备数据集')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mnist', 'cifar10', 'squad', 'all'],
        default='mnist',
        help='要准备的数据集 (默认: mnist)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='数据集存储目录 (默认: ./data)'
    )

    args = parser.parse_args()

    print("\n[INFO] ")
    print(f": {args.data_dir}")

    # 根据选择准备数据集
    if args.dataset == 'mnist' or args.dataset == 'all':
        prepare_mnist(args.data_dir)

    if args.dataset == 'cifar10' or args.dataset == 'all':
        prepare_cifar10(args.data_dir)

    if args.dataset == 'squad' or args.dataset == 'all':
        prepare_squad(args.data_dir)

    print("\n" + "="*50)
    print("[SUCCESS] ")
    print("="*50)


if __name__ == '__main__':
    main()
