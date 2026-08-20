"""Train CIFAR-10 ResNet-18 baseline checkpoint."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.model_factory import build_model_from_config
from src.models.dense_baseline import ModelTrainer
from src.utils.data_loader import cifar10_split_metadata, get_cifar10_loaders
from src.utils.device import configure_cuda, resolve_device
from src.utils.experiment_artifacts import load_config, set_cpu_threads, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 ResNet baseline")
    parser.add_argument("--config", default="configs/cifar_resnet_baseline_gpu.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(config["hardware"]["device"])
    set_seed(config["seed"], deterministic=bool(config["hardware"].get("deterministic", True)))
    set_cpu_threads(config["hardware"].get("cpu_threads"))
    configure_cuda(config)

    split_seed = config["dataset"].get("split_seed", config["seed"])
    train_loader, validation_loader = get_cifar10_loaders(
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        validation_fraction=config["dataset"].get("validation_fraction", 0.1),
        split_seed=split_seed,
        pin_memory=config["dataset"].get("pin_memory"),
        persistent_workers=bool(config["dataset"].get("persistent_workers", False)),
        prefetch_factor=config["dataset"].get("prefetch_factor"),
    )
    split = cifar10_split_metadata(train_loader, validation_loader, split_seed)
    print(f"[OK] CIFAR split hash: {split['split_hash']}")
    print(f"[OK] train={split['train_size']} validation={split['validation_size']}")

    model = build_model_from_config(config)
    print(f"[OK] parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    trainer = ModelTrainer(
        model=model,
        device=str(device),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    save_path = config["training"]["save_path"]
    epochs = int(config["training"]["epochs"])
    print(f"[INFO] Training {epochs} epochs on {device}")
    start = time.perf_counter()
    history = trainer.train(
        train_loader=train_loader,
        test_loader=validation_loader,
        epochs=epochs,
        verbose=True,
        save_path=save_path,
    )
    elapsed = time.perf_counter() - start
    print(f"[OK] Finished in {elapsed:.1f}s")
    print(f"[OK] Best validation acc: {max(history['test_acc']):.2f}%")
    print(f"[OK] Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
