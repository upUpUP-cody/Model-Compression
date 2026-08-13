"""Verify the local PyTorch CUDA environment before running a GPU study."""
from __future__ import annotations

import argparse

import torch

from src.models.dense_baseline import MLP
from src.utils.device import configure_cuda, device_metadata, resolve_device
from src.utils.precision import autocast_context, grad_scaler, resolve_precision


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CUDA device and AMP execution")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="fp16", choices=("fp16", "bf16"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    policy = configure_cuda({"seed": 42, "hardware": {"device": str(device)}})
    print(f"[OK] Device: {device}")
    print(f"[OK] PyTorch: {torch.__version__}")
    print(f"[OK] CUDA metadata: {device_metadata(device)}")
    print(f"[OK] Policy: {policy}")

    model = MLP(4, [4], 2, dropout_rate=0.0, use_batch_norm=False).to(device)
    inputs = torch.randn(8, 4, device=device)
    targets = torch.randint(0, 2, (8,), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = grad_scaler(device, precision)
    optimizer.zero_grad()
    with autocast_context(device, precision):
        loss = criterion(model(inputs), targets)
    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    print(f"[OK] AMP forward/backward: precision={precision} loss={loss.item():.6f}")
    print(f"[OK] Peak allocated bytes: {allocated}")


if __name__ == "__main__":
    main()
