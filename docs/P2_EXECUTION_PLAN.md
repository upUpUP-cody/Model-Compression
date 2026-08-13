# P2 执行计划：CIFAR-10 + ResNet GPU 扩展

> P2 阶段的详细执行手册。长期路线图见 [PROJECT_PLAN.md](../PROJECT_PLAN.md)，当前门禁状态见 [EXECUTION_PLAN.md](../EXECUTION_PLAN.md)。

## 前置条件（已完成）

- P0 搜索正确性、Pareto frontier、train/validation/test 隔离
- MNIST P1.2 GPU：smoke、formal、3 seed multiseed、6 档压缩率 sweep、聚合报告
- GPU 基础设施：`venv/`、`scripts/run_gpu.sh`、`scripts/check_gpu.py`、device/AMP/benchmark 工具

## 运行环境

```bash
cd /root/Model-Compression
source venv/bin/activate
export PYTHONPATH=/root/Model-Compression
# 或：./scripts/run_gpu.sh python ...
python scripts/check_gpu.py --device cuda:0 --precision bf16
```

## P2.1 CIFAR-10 数据与 ResNet 基线

**涉及文件：** `src/utils/data_loader.py`、`src/models/resnet_cifar.py`、`experiments/exp_cifar_baseline.py`、`configs/cifar_resnet_baseline_gpu.yaml`

**命令：**

```bash
./scripts/run_gpu.sh python experiments/exp_cifar_baseline.py --config configs/cifar_resnet_baseline_gpu.yaml
```

**输出：** `checkpoints/cifar_resnet18_baseline.pth`

**验收：** 同 `split_seed` 可复现 `split_hash`；训练脚本不使用 test loader。

**GPU 预计时长（RTX 4090）：** smoke 20 epoch 约 15–25 分钟；正式 100+ epoch 约 1–3 小时。

## P2.2 CNN 结构化剪枝

**涉及文件：** `src/pruning/cnn_dependency.py`、`src/pruning/cnn_structured_pruning.py`、`tests/test_cnn_structured_pruning.py`

在每个 ResNet BasicBlock 内剪枝 `conv1` 输出通道；同步 `bn1` 与 `conv2` 输入。Block 输入/输出宽度不变。

## P2.3 Wanda 与 CNN 搜索迁移

**涉及文件：** `src/pruning/pruning_backend.py`、`src/pruning/sensitivity.py`（Conv Wanda）、`src/autonomous_search.py`（`model.type`）

默认仍为 MLP；`model.type: resnet_cifar` 时选择 CNN backend。

## P2.4 CIFAR P1.2 GPU 对照协议

**涉及文件：** `src/experiments/cifar_p12_comparison.py`、`experiments/run_cifar_p12_comparison.py`、`experiments/run_cifar_p12_multiseed.py`、`configs/cifar_p12_gpu_*.yaml`

**输出根目录：** `results/cifar_p12_comparison_gpu_smoke/`、`results/cifar_p12_comparison_gpu/` 等（与 MNIST 分离）。

**工作流：**

```bash
# 1. 先训练基线 checkpoint
./scripts/run_gpu.sh python experiments/exp_cifar_baseline.py --config configs/cifar_resnet_baseline_gpu.yaml

# 2. Smoke
./scripts/run_gpu.sh python experiments/run_cifar_p12_comparison.py study \
  --config configs/cifar_p12_gpu_smoke.yaml \
  --checkpoint checkpoints/cifar_resnet18_baseline.pth

./scripts/run_gpu.sh python experiments/run_cifar_p12_comparison.py report-test \
  --config configs/cifar_p12_gpu_smoke.yaml \
  --checkpoint checkpoints/cifar_resnet18_baseline.pth \
  --study-dir <SMOKE_STUDY_DIR>

# 3. Formal / multiseed / sweep（smoke 通过后）
./scripts/run_gpu.sh python experiments/run_cifar_p12_multiseed.py \
  --config configs/cifar_p12_gpu_multiseed.yaml \
  --checkpoint checkpoints/cifar_resnet18_baseline.pth

./scripts/run_gpu.sh python experiments/run_cifar_p12_multiseed.py \
  --config configs/cifar_p12_gpu_sweep.yaml \
  --checkpoint checkpoints/cifar_resnet18_baseline.pth --sweep
```

**GPU 预计时长：**

| 步骤 | 时长 |
|------|------|
| CIFAR smoke | 10–20 分钟 |
| CIFAR formal（1 seed） | 20–40 分钟 |
| 3 seed multiseed | 1–2 小时 |
| 6 比率 × 3 seed sweep | 3–6 小时 |

## P2.5 GPU 吞吐与显存

manifest 的 `runtime.inference_benchmark` 包含 `peak_allocated_bytes`、延迟与吞吐（`src/utils/gpu_benchmark.py`）。

## P2.6 LoRA 恢复（Level 2）

**涉及文件：** `src/recovery/lora_recovery.py`、`tests/test_lora_recovery.py`

配置：`recovery.level: 2`、`lora_rank`、`lora_alpha`。

## P2.7 自蒸馏恢复（Level 3）

**涉及文件：** `src/recovery/self_distillation.py`、`tests/test_self_distillation.py`

教师 = 冻结 baseline；学生 = 剪枝后模型。

## P2.8 恢复消融

在固定 CIFAR 剪枝候选上对比 recovery level 1/2/3。

**命令：**

```bash
./scripts/run_gpu.sh python experiments/run_cifar_recovery_ablation.py \
  --config configs/cifar_recovery_ablation.yaml \
  --checkpoint checkpoints/cifar_resnet18_baseline.pth
```

**输出：** `results/cifar_recovery_ablation/`、`ABLATION_REPORT.md`

## P2.9 Qwen/SQuAD（仅规划，不实现）

- Transformer 剪枝单元（head/FFN）
- 指标：F1/EM
- 硬件：>=16GB VRAM
- **启动条件：** CIFAR P1.2 + recovery 消融通过审查

## 门禁

| 门禁 | 要求 |
|------|------|
| G1 | CIFAR baseline checkpoint + split metadata |
| G2 | CNN 剪枝测试通过 |
| G3 | MNIST 测试不退化 |
| G4 | CIFAR smoke 冻结 + report-test |
| G5 | multiseed/sweep 聚合 |
| G6 | LoRA/自蒸馏测试通过 |
| G7 | 消融报告 |
