# GPU 续跑工作流

本工作流面向已生成 `gpu_send/gpu_send/` P1.2 回传证据的同一台 RTX 4090 主机。该证据是 seed=42 的已完成单次参考结果，不是待补跑任务。

## GitHub 增量交付

GPU 主机只应拉取已批准的源码 commit：

```bat
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse HEAD
git status --short
```

- 将输出 SHA 与交付记录中的批准 SHA 比对。
- `gpu_send/gpu_send/` 是只读回传证据：不得覆盖、移动、修改或提交。
- 不得提交或回推 `results/`、`data/`、`venv/`、缓存、日志或实验生成 checkpoint。
- 只有本次 commit 修改了 P1.2 行为、协议、配置语义或基线 checkpoint 输入时，才可用新的输出根运行新的 P1.2 study；否则保留既有 evidence，不重跑它。

## 环境核验

保留已经验证的 CUDA 环境；仅在驱动、PyTorch 或 GPU 分配变化时才选择新的兼容 wheel，并记录完整安装命令和实际版本。

```bat
nvidia-smi
python -c "import torch, torchvision, torchaudio; print('torch=', torch.__version__); print('torchvision=', torchvision.__version__); print('torchaudio=', torchaudio.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available())"
python scripts/check_gpu.py --device cuda:0 --precision fp16
python scripts/check_gpu.py --device cuda:0 --precision bf16
```

CUDA、AMP forward/backward 或非零显存分配失败时停止。若 BF16 不支持，回传 capability 与错误输出；不得临时编辑 BF16 formal YAML 改成 FP16，必须先获得批准的新版本化 FP16 formal config。

## 新 P1.2 study（仅在输入或协议变更时）

先执行仅 train/validation 的 smoke study，再使用命令实际打印的新目录执行一次 frozen report：

```bat
python experiments/run_p12_comparison.py study --config configs/mnist_p12_gpu_smoke.yaml --checkpoint checkpoints/mnist_dense_baseline.pth
python experiments/run_p12_comparison.py report-test --config configs/mnist_p12_gpu_smoke.yaml --checkpoint checkpoints/mnist_dense_baseline.pth --study-dir <SMOKE_STUDY_DIR>
```

Smoke 通过后，针对新的 formal 目录执行同一流程：

```bat
python experiments/run_p12_comparison.py study --config configs/mnist_p12_gpu_study.yaml --checkpoint checkpoints/mnist_dense_baseline.pth
python experiments/run_p12_comparison.py report-test --config configs/mnist_p12_gpu_study.yaml --checkpoint checkpoints/mnist_dense_baseline.pth --study-dir <FORMAL_STUDY_DIR>
```

`report-test` 只允许用于新的、未生成 final report 的 frozen study。它会在加载 test 数据前核验：P1.2 protocol、六种固定方法、config/source checkpoint/Git SHA、split metadata、CUDA identity（GPU profile）、study 内 checkpoint 与 SHA-256。

预计超过五分钟的新实验启动后，必须立即记录实验名称、基于首轮实测的时长、命令和实际输出目录。回传 evidence 时独立打包 manifest、resolved config、comparison JSON/CSV、summary、final report、六个 checkpoint 的 digest/安全位置和完整日志；不得推送至 GitHub。

## 后续阶段

P1.2 seed=42 不构成统计结论；MNIST multiseed/sweep 聚合已完成。

**P2 CIFAR GPU 工作流**（详见 [docs/P2_EXECUTION_PLAN.md](P2_EXECUTION_PLAN.md)）：

```bash
# 环境
source venv/bin/activate
export PYTHONPATH=/root/Model-Compression

# 1. 训练 CIFAR ResNet 基线
./scripts/run_gpu.sh python experiments/exp_cifar_baseline.py --config configs/cifar_resnet_baseline_gpu.yaml

# 2. CIFAR P1.2 smoke + report-test
./scripts/run_gpu.sh python experiments/run_cifar_p12_comparison.py study \
  --config configs/cifar_p12_gpu_smoke.yaml \
  --checkpoint checkpoints/cifar_resnet18_baseline.pth

./scripts/run_gpu.sh python experiments/run_cifar_p12_comparison.py report-test \
  --config configs/cifar_p12_gpu_smoke.yaml \
  --checkpoint checkpoints/cifar_resnet18_baseline.pth \
  --study-dir <SMOKE_STUDY_DIR>

# 3. multiseed / sweep
./scripts/run_gpu.sh python experiments/run_cifar_p12_multiseed.py \
  --config configs/cifar_p12_gpu_multiseed.yaml \
  --checkpoint checkpoints/cifar_resnet18_baseline.pth

# 4. 恢复消融
./scripts/run_gpu.sh python experiments/run_cifar_recovery_ablation.py \
  --config configs/cifar_recovery_ablation.yaml \
  --checkpoint checkpoints/cifar_resnet18_baseline.pth
```

CIFAR 输出根目录与 MNIST 分离（`results/cifar_p12_comparison_gpu_*`）。预计超过五分钟的实验启动后须立即报告实验名、时长与输出目录。

## MNIST 归档参考

以下为已完成 MNIST P1.2 GPU 流程，供对照，不应覆盖已冻结 study 目录。
