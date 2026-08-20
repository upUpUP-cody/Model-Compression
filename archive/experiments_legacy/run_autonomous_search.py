"""Run the CPU-first MNIST autonomous structured-pruning MVP."""
import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autonomous_search import AutonomousSearch
from src.controller.heuristic_controller import HeuristicController
from src.evaluation.frontier import ParetoFrontier
from src.models.dense_baseline import MLP
from src.utils.data_loader import get_mnist_loaders, mnist_split_metadata
from src.utils.experiment_artifacts import (
    RunArtifacts,
    config_hash,
    file_sha256,
    git_sha,
    load_config,
    runtime_metadata,
    set_cpu_threads,
    set_seed,
)
from src.utils.search_visualization import plot_search_history


def load_baseline(model: MLP, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU autonomous structured-pruning search")
    parser.add_argument(
        "--config",
        default="configs/mnist_mlp_autonomous_cpu.yaml",
        help="Path to a CPU MVP YAML configuration",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/mnist_dense_baseline.pth",
        help="Path to the dense baseline checkpoint",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    set_cpu_threads(config["hardware"].get("cpu_threads"))
    checkpoint_source = Path(args.checkpoint)
    model_config = config["model"]
    model = MLP(**model_config)
    load_baseline(model, checkpoint_source)
    split_seed = config["dataset"].get("split_seed", config["seed"])
    train_loader, validation_loader, test_loader = get_mnist_loaders(
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        validation_fraction=config["dataset"].get("validation_fraction", 0.1),
        split_seed=split_seed,
        return_test=True,
    )
    split = mnist_split_metadata(train_loader, validation_loader, split_seed)

    run_id = f"{config.get('run_label', 'mnist')}_{config_hash(config)[:12]}"
    artifacts = RunArtifacts(config["logging"]["output_root"], run_id=run_id)
    artifacts.save_config(config)
    controller = HeuristicController(**config["controller"])
    frontier = ParetoFrontier()
    search = AutonomousSearch(controller=controller)
    accepted_model, history = search.run(
        model,
        train_loader,
        validation_loader,
        max_iterations=config["search"]["max_iterations"],
        candidate_ratios=config["search"]["candidate_ratios"],
        candidates_per_round=config["search"]["candidates_per_round"],
        cheap_eval_samples=config["search"]["cheap_eval_samples"],
        recovery_epochs=config["recovery"]["epochs"],
        recovery_learning_rate=config["recovery"]["learning_rate"],
        device=config["hardware"]["device"],
        enable_two_layer_candidates=config["search"].get("enable_two_layer_candidates", False),
        recovery_top_k=config["search"].get("recovery_top_k", 1),
        frontier_archive=frontier,
    )
    for event in history.events:
        artifacts.append_event(event)
    artifacts.save_history_csv(history.events)
    frontier_json, frontier_csv = artifacts.save_frontier(frontier)
    checkpoint_path = artifacts.save_checkpoint(accepted_model)
    test_report = search.evaluator(accepted_model, test_loader, config["hardware"]["device"])
    performance = artifacts.measure_inference(accepted_model, test_loader, config["hardware"]["device"])
    manifest_path = artifacts.save_manifest({
        "config_hash": config_hash(config),
        "git_sha": git_sha(PROJECT_ROOT),
        "checkpoint_source": str(checkpoint_source),
        "checkpoint_source_sha256": file_sha256(checkpoint_source),
        "accepted_checkpoint": str(checkpoint_path),
        "accepted_checkpoint_sha256": file_sha256(checkpoint_path),
        "command": " ".join(sys.argv),
        "seed": config["seed"],
        "split": split,
        "runtime": runtime_metadata(),
        "frontier_json": str(frontier_json),
        "frontier_csv": str(frontier_csv),
        "test_report": test_report,
        "performance": performance,
    })
    summary = history.to_dict()
    summary.update({
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "frontier": str(frontier_json),
        "test_report": test_report,
        "performance": performance,
        "stop_reason": history.events[-1].get("reason") if history.events else "no_events",
        "no_candidate_accepted": not any(event.get("final_action") == "accept" for event in history.events),
    })
    artifacts.save_summary(summary)
    accepted_events = [event for event in history.events if event.get("final_action") == "accept"]
    if accepted_events:
        plot_search_history(accepted_events, artifacts.run_dir / "search_history.png")
    print(f"[OK] Search artifacts: {artifacts.run_dir}")


if __name__ == "__main__":
    main()
