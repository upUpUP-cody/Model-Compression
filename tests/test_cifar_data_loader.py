import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.utils.data_loader import cifar10_split_metadata


def _loaders_from_seed(split_seed: int):
    full = TensorDataset(torch.zeros(500), torch.zeros(500, dtype=torch.long))
    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(len(full), generator=generator).tolist()
    validation_indices = indices[:50]
    train_indices = indices[50:]
    train_loader = DataLoader(Subset(full, train_indices), batch_size=16)
    validation_loader = DataLoader(Subset(full, validation_indices), batch_size=16)
    return train_loader, validation_loader


def test_cifar10_split_metadata_is_reproducible():
    train_a, val_a = _loaders_from_seed(7)
    train_b, val_b = _loaders_from_seed(7)
    meta_a = cifar10_split_metadata(train_a, val_a, 7)
    meta_b = cifar10_split_metadata(train_b, val_b, 7)
    assert meta_a == meta_b
    assert meta_a["train_size"] == 450
    assert meta_a["validation_size"] == 50


def test_cifar10_split_metadata_differs_by_seed():
    train_a, val_a = _loaders_from_seed(7)
    train_b, val_b = _loaders_from_seed(8)
    assert cifar10_split_metadata(train_a, val_a, 7)["split_hash"] != cifar10_split_metadata(
        train_b, val_b, 8
    )["split_hash"]
