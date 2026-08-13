import torch

from src.models.dense_baseline import MLP
from src.models.resnet_cifar import resnet18_cifar
from src.pruning.pruning_backend import CnnBackend, MlpBackend, resolve_pruning_backend


def test_mlp_backend_prunable_layers():
    model = MLP(input_dim=4, hidden_dims=[8, 4], num_classes=2, use_batch_norm=False)
    backend = MlpBackend(model)
    assert backend.prunable_layer_names() == ["features.0", "features.3"]


def test_cnn_backend_prunable_layers():
    model = resnet18_cifar(base_width=8)
    backend = CnnBackend(model)
    assert "layer1.0.conv1" in backend.prunable_layer_names()


def test_resolve_backend_from_model_type():
    model = resnet18_cifar(base_width=8)
    backend = resolve_pruning_backend(model, "resnet_cifar")
    assert isinstance(backend, CnnBackend)
    pruned = backend.create_pruned_model({"layer1.0.conv1": 0.5})
    assert sum(parameter.numel() for parameter in pruned.parameters()) < model.get_num_parameters()
