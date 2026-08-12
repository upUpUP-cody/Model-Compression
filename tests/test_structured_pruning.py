import copy
import json

import pytest
import torch
import torch.nn as nn

from src.models.dense_baseline import MLP
from src.pruning.structured_pruning import StructuredPruning


def make_model(use_batch_norm=True, dropout_rate=0.0):
    torch.manual_seed(7)
    return MLP(
        input_dim=8,
        hidden_dims=[6, 5, 4],
        num_classes=3,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
    )


def clone_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_state_equal(model, state):
    current_state = model.state_dict()
    assert current_state.keys() == state.keys()
    for name, expected in state.items():
        assert torch.equal(current_state[name], expected), name


def test_prune_linear_block_updates_batch_norm_and_connected_linear():
    model = make_model()
    pruner = StructuredPruning(model)
    indices = [1, 4, 5]

    original_linear_weight = model.features[0].weight.detach().clone()
    original_linear_bias = model.features[0].bias.detach().clone()
    original_bn_weight = model.features[1].weight.detach().clone()
    original_bn_running_mean = model.features[1].running_mean.detach().clone()
    original_next_weight = model.features[3].weight.detach().clone()

    pruner.prune_linear_block("features.0", indices, "features.3")

    assert model.features[0].out_features == 3
    assert model.features[1].num_features == 3
    assert model.features[3].in_features == 3
    assert torch.equal(model.features[0].weight, original_linear_weight[indices])
    assert torch.equal(model.features[0].bias, original_linear_bias[indices])
    assert torch.equal(model.features[1].weight, original_bn_weight[indices])
    assert torch.equal(model.features[1].running_mean, original_bn_running_mean[indices])
    assert torch.equal(model.features[3].weight, original_next_weight[:, indices])
    assert pruner.pruning_masks["features.0"] == {
        "type": "neuron",
        "indices": indices,
        "original_size": 6,
        "retained_size": 3,
    }
    json.dumps(pruner.pruning_masks)


def test_prune_discovers_next_linear_without_batch_norm():
    model = make_model(use_batch_norm=False, dropout_rate=0.2)
    pruner = StructuredPruning(model)

    pruner.prune_mlp_neurons("features.0", [0, 2, 5])

    assert model.features[0].out_features == 3
    assert model.features[3].in_features == 3
    assert model(torch.randn(4, 8)).shape == (4, 3)


def test_multilayer_pruning_supports_backward_and_optimizer_step():
    model = make_model()
    pruner = StructuredPruning(model)
    original_parameter_count = pruner.get_structural_parameter_count()

    pruner.prune_by_layer({"features.0": 0.5, "features.3": 0.4, "features.6": 0.25})

    assert model.classifier.out_features == 3
    assert model.classifier.in_features == model.features[6].out_features
    assert pruner.get_structural_parameter_count() < original_parameter_count

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    logits = model(torch.randn(6, 8))
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2, 0, 1, 2]))
    loss.backward()
    optimizer.step()


def test_create_pruned_model_preserves_source_model():
    model = make_model()
    source_state = clone_state(model)
    source_shapes = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}

    pruned_model = StructuredPruning(model).create_pruned_model({"features.0": 0.5})

    assert pruned_model is not model
    assert pruned_model.features[0].out_features == 3
    assert model.features[0].out_features == 6
    assert_state_equal(model, source_state)
    assert {name: tuple(parameter.shape) for name, parameter in model.named_parameters()} == source_shapes


def test_ratio_selection_is_deterministic_and_zero_is_noop():
    model = make_model()
    pruner = StructuredPruning(model)
    scores = torch.tensor([1.0, 4.0, 4.0, 2.0, 3.0, 0.0])

    assert pruner.prune_mlp_by_ratio("features.0", 0.5, scores) == [1, 2, 4]
    assert pruner.prune_mlp_by_ratio("features.0", 0.0, scores) == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "operation",
    [
        lambda pruner: pruner.prune_mlp_neurons("classifier", [0]),
        lambda pruner: pruner.prune_mlp_neurons("missing.layer", [0]),
        lambda pruner: pruner.prune_mlp_neurons("features.1", [0]),
        lambda pruner: pruner.prune_mlp_by_ratio("features.0", -0.1),
        lambda pruner: pruner.prune_mlp_by_ratio("features.0", 1.0),
        lambda pruner: pruner.prune_mlp_by_ratio("features.0", float("nan")),
        lambda pruner: pruner.prune_mlp_neurons("features.0", []),
        lambda pruner: pruner.prune_mlp_neurons("features.0", [1, 1]),
        lambda pruner: pruner.prune_mlp_neurons("features.0", [6]),
        lambda pruner: pruner.prune_mlp_neurons("features.0", ["1"]),
        lambda pruner: pruner.prune_mlp_by_ratio("features.0", 0.5, torch.ones(5)),
        lambda pruner: pruner.prune_mlp_by_ratio(
            "features.0", 0.5, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, float("inf")])
        ),
        lambda pruner: pruner.prune_linear_block("features.0", [0, 1], "features.6"),
    ],
)
def test_invalid_requests_raise_without_mutating_model(operation):
    model = make_model()
    state = clone_state(model)

    with pytest.raises(ValueError):
        operation(StructuredPruning(model))

    assert_state_equal(model, state)


def test_get_model_info_reports_structural_parameter_count():
    model = make_model()
    pruner = StructuredPruning(model)
    before = pruner.get_model_info()

    pruner.prune_mlp_neurons("features.0", [0, 1, 2])
    after = pruner.get_model_info()

    assert after["total_params"] == after["structural_parameter_count"]
    assert after["total_params"] < before["total_params"]
    assert 0.0 <= after["sparsity"] <= 1.0
