# Environment

## Current CPU development host

The current workstation is used for CPU regression tests and code development. CUDA validation has not been performed here and must not be reported as complete.

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

## CUDA host setup

On the new NVIDIA GPU host, first confirm the driver:

```bash
nvidia-smi
```

Create an isolated environment, install the generic project dependencies, then install the PyTorch CUDA wheel that matches the driver and PyTorch compatibility matrix. Example only for CUDA 12.6:

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python scripts/check_gpu.py --device cuda:0 --precision fp16
```

`requirements-gpu.txt` records the required installation sequence but does not pin a wheel index because the correct index depends on the target driver.

See [docs/GPU_WORKFLOW.md](docs/GPU_WORKFLOW.md) for the required smoke, frozen report-test, and formal-study sequence. Do not run a GPU config on CPU as a fallback.

## Core constraints

- Python `print()` messages must use ASCII-only text because Windows GBK terminals can fail on emoji.
- Keep datasets, checkpoint files, results, caches, and local `.claude/settings.json` outside commits.
- GPU smoke and formal study use independent output roots and must not overwrite CPU artifacts.
