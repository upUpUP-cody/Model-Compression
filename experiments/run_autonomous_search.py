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
from src.models.dense_baseline import MLP
from src.utils.data_loader import get_mnist_loaders
from src.utils.experiment_artifacts import RunArtifacts, load_config, set_seed
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
    model_config = config["model"]
    model = MLP(**model_config)
    load_baseline(model, Path(args.checkpoint))
    train_loader, validation_loader = get_mnist_loaders(
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )

    artifacts = RunArtifacts(config["logging"]["output_root"])
    artifacts.save_config(config)
    controller = HeuristicController(**config["controller"])
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
    )
    for event in history.events:
        artifacts.append_event(event)
    artifacts.save_history_csv(history.events)
    checkpoint_path = artifacts.save_checkpoint(accepted_model)
    summary = history.to_dict()
    summary["checkpoint"] = str(checkpoint_path)
    artifacts.save_summary(summary)
    accepted_events = [event for event in history.events if event.get("final_action") == "accept"]
    if accepted_events:
        plot_search_history(accepted_events, artifacts.run_dir / "search_history.png")
    print(f"[OK] Search artifacts: {artifacts.run_dir}")


if __name__ == "__main__":
    main()
