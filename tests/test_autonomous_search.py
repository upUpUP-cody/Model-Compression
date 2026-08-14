import copy
import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.autonomous_search import AutonomousSearch, CandidateSpec, candidate_fingerprint
from src.controller.heuristic_controller import HeuristicController
from src.evaluation.cheap_critic import CheapCriticResult
from src.evaluation.frontier import ParetoFrontier
from src.models.dense_baseline import MLP
from src.models.resnet_cifar import resnet18_cifar
from src.pruning.pruning_backend import CnnBackend


class FixedCritic:
    def evaluate(self, model, dataloader, max_samples, device):
        return CheapCriticResult(
            loss=0.1,
            accuracy=90.0,
            samples=min(max_samples, len(dataloader.dataset)),
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            nonzero_parameter_count=sum(
                (parameter.detach() != 0).sum().item() for parameter in model.parameters()
            ),
            elapsed_seconds=0.0,
        )


def make_model():
    torch.manual_seed(13)
    return MLP(
        input_dim=4,
        hidden_dims=[4, 3],
        num_classes=2,
        dropout_rate=0.0,
        use_batch_norm=False,
    )


def make_loader():
    torch.manual_seed(3)
    return DataLoader(
        TensorDataset(torch.randn(12, 4), torch.randint(0, 2, (12,))),
        batch_size=4,
        shuffle=False,
    )


def evaluator(model, loader, device):
    return {"loss": 0.1, "accuracy": 90.0, "samples": len(loader.dataset)}


def importance(model, loader, device):
    return {"features.0": torch.tensor([4.0, 3.0, 2.0, 1.0])}


def recovery(model, train_loader, validation_loader, **kwargs):
    return model, {"best_validation_accuracy": 90.0}


def test_candidate_fingerprint_is_order_independent():
    assert candidate_fingerprint({"features.3": 0.5, "features.0": 0.3}) == candidate_fingerprint(
        {"features.0": 0.3, "features.3": 0.5}
    )


def test_search_accepts_pruned_copy_without_mutating_source_model():
    source = make_model()
    source_state = copy.deepcopy(source.state_dict())
    search = AutonomousSearch(
        controller=HeuristicController(max_accuracy_drop_points=5.0),
        critic=FixedCritic(),
        evaluator=evaluator,
        importance_fn=importance,
        recovery_fn=recovery,
    )

    accepted, history = search.run(
        source,
        make_loader(),
        make_loader(),
        max_iterations=1,
        candidate_ratios=(0.5,),
        candidates_per_round=1,
        cheap_eval_samples=5,
        recovery_epochs=0,
    )

    assert accepted is not source
    assert accepted.features[0].out_features == 2
    assert source.features[0].out_features == 4
    for name, expected in source_state.items():
        assert torch.equal(source.state_dict()[name], expected), name
    assert history.events[0]["final_action"] == "accept"
    json.dumps(history.to_dict())


def test_full_validation_rejection_leaves_accepted_model_unchanged():
    source = make_model()
    source_state = copy.deepcopy(source.state_dict())
    evaluations = iter(
        [
            {"loss": 0.1, "accuracy": 90.0, "samples": 12},
            {"loss": 0.5, "accuracy": 80.0, "samples": 12},
        ]
    )
    search = AutonomousSearch(
        controller=HeuristicController(max_accuracy_drop_points=1.0),
        critic=FixedCritic(),
        evaluator=lambda *args: next(evaluations),
        importance_fn=importance,
        recovery_fn=recovery,
    )

    accepted, history = search.run(
        source,
        make_loader(),
        make_loader(),
        max_iterations=1,
        candidate_ratios=(0.5,),
        candidates_per_round=1,
        cheap_eval_samples=5,
        recovery_epochs=0,
    )

    assert accepted.features[0].out_features == 4
    assert history.events[0]["final_action"] == "regrow"
    assert history.events[0]["final_reason"] == "capability_gap_exceeded"
    for name, expected in source_state.items():
        assert torch.equal(source.state_dict()[name], expected), name


class LowCheapHighRecoveryCritic:
    def evaluate(self, model, dataloader, max_samples, device):
        return CheapCriticResult(
            loss=3.0,
            accuracy=20.0,
            samples=min(max_samples, len(dataloader.dataset)),
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            nonzero_parameter_count=sum(
                (parameter.detach() != 0).sum().item() for parameter in model.parameters()
            ),
            elapsed_seconds=0.0,
        )


def test_search_recovers_before_capability_gate_and_can_accept():
    """Cheap Critic may look terrible; post-recovery accuracy decides accept/regrow."""
    source = make_model()
    recovery_calls = {"count": 0}

    def tracking_recovery(model, train_loader, validation_loader, **kwargs):
        recovery_calls["count"] += 1
        return model, {"best_validation_accuracy": 90.0}

    search = AutonomousSearch(
        controller=HeuristicController(max_accuracy_drop_points=2.0),
        critic=LowCheapHighRecoveryCritic(),
        evaluator=evaluator,
        importance_fn=importance,
        recovery_fn=tracking_recovery,
    )
    accepted, history = search.run(
        source,
        make_loader(),
        make_loader(),
        max_iterations=1,
        candidate_ratios=(0.5,),
        candidates_per_round=1,
        cheap_eval_samples=5,
        recovery_epochs=1,
    )

    assert recovery_calls["count"] == 1
    assert history.events[0]["cheap_critic"]["accuracy"] == 20.0
    assert history.events[0]["validation"]["accuracy"] == 90.0
    assert history.events[0]["final_action"] == "accept"
    assert accepted.features[0].out_features == 2
    assert sum(p.numel() for p in accepted.parameters()) < sum(p.numel() for p in source.parameters())


def test_search_skips_previously_attempted_configurations():
    source = make_model()
    search = AutonomousSearch(
        controller=HeuristicController(),
        critic=FixedCritic(),
        evaluator=evaluator,
        importance_fn=importance,
        recovery_fn=recovery,
    )

    _, history = search.run(
        source,
        make_loader(),
        make_loader(),
        max_iterations=3,
        candidate_ratios=(0.5,),
        candidates_per_round=1,
        cheap_eval_samples=5,
        recovery_epochs=0,
    )

    assert len(history.events) == 2
    assert history.events[-1]["action"] == "stop"
    assert history.events[-1]["reason"] == "no_new_candidates"


def test_search_physically_uses_wanda_indices_and_records_candidate_spec():
    source = make_model()
    with torch.no_grad():
        source.features[0].weight.copy_(
            torch.tensor(
                [
                    [10.0, 0.0, 0.0, 0.0],
                    [9.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.0],
                ]
            )
        )
    original_weight = source.features[0].weight.detach().clone()

    search = AutonomousSearch(
        controller=HeuristicController(max_accuracy_drop_points=5.0),
        critic=FixedCritic(),
        evaluator=evaluator,
        importance_fn=lambda *_args: {"features.0": torch.tensor([1.0, 2.0, 9.0, 8.0])},
        recovery_fn=recovery,
    )
    accepted, history = search.run(
        source,
        make_loader(),
        make_loader(),
        max_iterations=1,
        candidate_ratios=(0.5,),
        candidates_per_round=1,
        cheap_eval_samples=5,
        recovery_epochs=0,
    )

    assert torch.equal(accepted.features[0].weight, original_weight[[2, 3]])
    event = history.events[0]
    assert event["candidate_spec"]["layer_keep_indices"] == {"features.0": [2, 3]}
    assert event["importance_method"] == "wanda"
    assert event["actual_parameter_count"] == sum(
        parameter.numel() for parameter in accepted.parameters()
    )
    json.dumps(history.to_dict())


def test_search_records_all_full_validation_results_in_frontier_archive():
    source = make_model()
    archive = ParetoFrontier()
    search = AutonomousSearch(
        controller=HeuristicController(max_accuracy_drop_points=5.0),
        critic=FixedCritic(),
        evaluator=evaluator,
        importance_fn=importance,
        recovery_fn=recovery,
    )

    _, history = search.run(
        source,
        make_loader(),
        make_loader(),
        max_iterations=1,
        candidate_ratios=(0.5,),
        candidates_per_round=1,
        cheap_eval_samples=5,
        recovery_epochs=0,
        frontier_archive=archive,
    )

    assert len(archive) == 1
    frontier_point = archive.points[0]
    assert frontier_point.validation_accuracy == 90.0
    assert frontier_point.parameter_count == history.accepted_parameter_count
    assert frontier_point.compression_ratio == source_parameter_count(source) / history.accepted_parameter_count
    assert history.frontier_points == archive.to_dict()["points"]


def source_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def test_candidate_spec_is_immutable_and_fingerprint_includes_indices():
    first = CandidateSpec.create(
        {"features.0": 0.5},
        {"features.0": [1, 2]},
        "wanda",
        parameter_count=10,
        parent_parameter_count=20,
    )
    second = CandidateSpec.create(
        {"features.0": 0.5},
        {"features.0": [0, 2]},
        "wanda",
        parameter_count=10,
        parent_parameter_count=20,
    )

    assert first.fingerprint != second.fingerprint
    with pytest.raises(Exception):
        first.parameter_count = 9
    assert json.loads(json.dumps(first.to_dict())) == first.to_dict()


def test_cnn_search_proposes_uniform_all_layer_candidate_first():
    model = resnet18_cifar(base_width=16)
    backend = CnnBackend(model)
    importance = {
        name: torch.arange(backend.output_size(name), 0, -1, dtype=torch.float32)
        for name in backend.prunable_layer_names()
    }
    candidates = AutonomousSearch._generate_candidates(
        model,
        importance,
        ratios=(0.5,),
        multiplier=1.0,
        limit=1,
        attempted=set(),
        backend=backend,
    )
    assert len(candidates) == 1
    assert set(candidates[0].ratios_dict()) == set(backend.prunable_layer_names())
    assert all(ratio == 0.5 for ratio in candidates[0].ratios_dict().values())
