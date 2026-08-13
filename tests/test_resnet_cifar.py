import torch

from src.models.resnet_cifar import resnet18_cifar, resnet18_cifar_small


def test_resnet18_cifar_forward_shape():
    model = resnet18_cifar()
    output = model(torch.randn(4, 3, 32, 32))
    assert output.shape == (4, 10)


def test_resnet18_cifar_small_has_fewer_parameters():
    full = resnet18_cifar()
    small = resnet18_cifar_small()
    assert small.get_num_parameters() < full.get_num_parameters()
