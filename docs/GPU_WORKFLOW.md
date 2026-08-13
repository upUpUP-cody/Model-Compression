# GPU Workflow

This workflow prepares and runs the MNIST P1.2 comparison on a CUDA host. CUDA smoke and study results are pending until they are run on the target GPU environment.

## 1. Prepare the host

Use a supported NVIDIA driver and verify that the GPU is visible:

```bash
nvidia-smi
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Choose the PyTorch wheel index matching the installed driver and the PyTorch compatibility table. The CUDA toolkit is not normally required when using the official wheel, but the NVIDIA driver is required.

## 2. Run the environment check

```bash
python scripts/check_gpu.py --device cuda:0 --precision fp16
```

The output must report the selected GPU, CUDA and cuDNN metadata, deterministic policy, AMP forward/backward, and peak allocated memory. A CUDA-unavailable error is a failure to resolve before proceeding; it is never permission to run the GPU profile on CPU.

## 3. Run the GPU smoke study

```bash
python experiments/run_p12_comparison.py study --config configs/mnist_p12_gpu_smoke.yaml --checkpoint checkpoints/mnist_dense_baseline.pth
```

Expected output root: `results/p12_comparison_gpu_smoke/`. Confirm that the generated `manifest.json` has `selection_frozen: true`, exactly six comparison records, a CUDA runtime block, checkpoint hashes, and no `final_test_report.json`.

## 4. Run the frozen smoke test report

Substitute the study directory printed by the smoke command:

```bash
python experiments/run_p12_comparison.py report-test --config configs/mnist_p12_gpu_smoke.yaml --checkpoint checkpoints/mnist_dense_baseline.pth --study-dir results/p12_comparison_gpu_smoke/p12_comparison_mnist_p12_gpu_smoke_HASH
```

This is the first point at which the official test dataset is loaded. Confirm the final report has six methods and references only frozen checkpoints.

## 5. Run the formal GPU study

Only after the smoke study and frozen report pass:

```bash
python experiments/run_p12_comparison.py study --config configs/mnist_p12_gpu_study.yaml --checkpoint checkpoints/mnist_dense_baseline.pth
```

Expected output root: `results/p12_comparison_gpu/`. This is a separate result namespace and must not replace any CPU artifacts. The study can take several minutes depending on GPU speed, data access, and recovery epochs.

After the formal study freezes successfully, run `report-test` with the same formal GPU config and its generated study directory.

## Troubleshooting

- CUDA unavailable: reinstall a PyTorch wheel matching the NVIDIA driver and confirm `nvidia-smi` works in the same shell.
- CUDA out of memory: reduce `dataset.batch_size`; retain `num_workers: 0` for the smoke profile.
- Windows worker errors: use `num_workers: 0`, `persistent_workers: false`, and omit `prefetch_factor`.
- bf16 unsupported: change `hardware.precision` to `fp16` and retain `mixed_precision: true`.
- Reproducibility-oriented runs: use `deterministic: true`, `cudnn_benchmark: false`, and `tf32: false`.

Do not commit `results/`, datasets, caches, or checkpoints. Preserve the returned study path and its `manifest.json` when reporting results from the GPU host.
