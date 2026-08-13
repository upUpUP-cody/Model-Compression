import torch

from src.models.resnet_cifar import resnet18_cifar
from src.pruning.cnn_structured_pruning import CnnStructuredPruning


def test_prune_basic_block_keeps_forward_shape():
    model = resnet18_cifar(base_width=16)
    pruner = CnnStructuredPruning(model)
    keep = list(range(pruner._get_layer("layer1.0.conv1").out_channels // 2))
    pruner.prune_conv_block("layer1.0.conv1", keep)
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)
    assert pruner.get_structural_parameter_count() < resnet18_cifar(base_width=16).get_num_parameters()


def test_create_pruned_model_by_indices_is_independent():
    baseline = resnet18_cifar(base_width=16)
    pruner = CnnStructuredPruning(baseline)
    keep_indices = {
        "layer1.0.conv1": [0],
        "layer2.0.conv1": [0, 1],
    }
    pruned = pruner.create_pruned_model_by_indices(keep_indices)
    assert pruned(torch.randn(1, 3, 32, 32)).shape == (1, 10)
    assert baseline(torch.randn(1, 3, 32, 32)).shape == (1, 10)
