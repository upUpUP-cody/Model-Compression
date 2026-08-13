# GPU 机器接管执行手册

本文档用于将 MNIST P1.2 GPU 实验交接给另一台 NVIDIA CUDA 机器的操作者。所有命令均需在仓库根目录执行。当前 CPU 主机没有产生任何真实 CUDA 结果，所有 GPU 证据必须由目标 GPU 主机实际生成。

## 0. 执行顺序与硬性规则

必须严格按以下顺序执行：

1. 核验传输的仓库、checkpoint 和 NVIDIA 环境。
2. 运行 CUDA 与 AMP 自检。
3. 运行仅使用 train/validation 的 GPU smoke study。
4. 核验 smoke study 已冻结，再运行其冻结 test report。
5. 运行正式 GPU study。
6. 核验正式 study 已冻结，再运行其冻结 test report。
7. 保留并回传完整证据包。

以下规则不可违反：

- 不得在 CPU 上运行 GPU profile，也不得启用 CPU fallback。
- 未经研究负责人批准，不得修改 GPU YAML、seed、方法列表、checkpoint 或输出目录。
- 不得对未冻结、旧的或不同的 study 目录运行 `report-test`。
- `study` 只可使用 train 和 validation 数据做选择；仅当冻结校验通过后，`report-test` 才可加载官方 MNIST test 数据。
- GPU 输出必须与已有 CPU 结果分离。
- 不得提交 `results/`、`data/`、虚拟环境、缓存或 checkpoint。
- 每条命令打印出的 study 目录必须原样保存；不得猜测或复用旧实验目录。

### 交接身份信息

启动前填好并随证据包一起回传：

```text
批准的仓库 Git SHA: <40 位 SHA>
基线 checkpoint SHA-256: <SHA-256>
传输压缩包 SHA-256: <SHA-256 或不适用>
目标设备: cuda:0
操作者: <姓名或标识>
主机: <机器标识>
启动时间: <本地时间>
```

Git revision 或基线 checkpoint 发生变化即代表实验输入变化，必须重新获得批准后从头执行。

## 1. 核验传输内容

先确认所需文件齐全：

```bat
if not exist configs\mnist_p12_gpu_smoke.yaml exit /b 1
if not exist configs\mnist_p12_gpu_study.yaml exit /b 1
if not exist experiments\run_p12_comparison.py exit /b 1
if not exist scripts\check_gpu.py exit /b 1
if not exist src exit /b 1
if not exist requirements.txt exit /b 1
if not exist checkpoints\mnist_dense_baseline.pth exit /b 1
```

记录源码、环境和 checkpoint 身份：

```bat
git rev-parse HEAD
git status --short
python --version
nvidia-smi
certutil -hashfile checkpoints\mnist_dense_baseline.pth SHA256
```

仅在以下条件全部满足时通过此 Gate：

- `git rev-parse HEAD` 与批准的 Git SHA 一致。
- `git status --short` 没有源码、配置、脚本或基线 checkpoint 的变动。
- `nvidia-smi` 成功，并显示目标 NVIDIA GPU。
- checkpoint SHA-256 与批准的 digest 一致。
- 所需文件均存在。

后续出现 `venv`、`data`、`results` 和缓存文件属于正常生成物，但不得加入 Git。此 Gate 失败时必须停止，修正传输内容或 GPU 主机分配后再继续。

## 2. 创建 CUDA 环境

在仓库目录创建隔离环境：

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

安装与 NVIDIA 驱动兼容的 CUDA PyTorch wheel：

```bat
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

`cu126` 是当前仓库文档中的示例，不保证适配所有驱动。若本机驱动需要其他受官方支持的 PyTorch CUDA wheel，请使用兼容的官方 index，并记录完整安装命令、index 和实际版本。不得通过修改实验 YAML 来掩盖安装问题。

记录安装后的运行时信息：

```bat
python -c "import torch, torchvision, torchaudio; print('torch=', torch.__version__); print('torchvision=', torchvision.__version__); print('torchaudio=', torchaudio.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
```

输出必须包含 `cuda_available=True` 且 `device_count` 至少为 1，否则不得继续。

## 3. CUDA 与 AMP 硬门槛

运行仓库自检并保存完整控制台输出：

```bat
python scripts/check_gpu.py --device cuda:0 --precision fp16
```

自检必须成功报告：

- 解析出的设备为 `cuda:0`。
- PyTorch 版本。
- CUDA 与 cuDNN runtime metadata。
- GPU 名称、capability、设备数量和显存。
- 已配置的 deterministic/CUDA policy。
- FP16 AMP forward 与 backward。
- 非零的 GPU peak allocated memory。

出现 `CUDA device requested but CUDA is unavailable`、`cuda:0` 不可用、tensor/model transfer 失败或 AMP backward 失败时，必须停止。请修复驱动、设备分配或 PyTorch wheel 后重跑本节；绝不可改为 CPU 执行。

正式 profile 默认使用 BF16，因此还应执行：

```bat
python scripts/check_gpu.py --device cuda:0 --precision bf16
```

如果 BF16 不受支持，不得自行将正式配置改成 FP16。必须将 GPU capability 和错误输出回传给研究负责人；需要另行批准并版本化 FP16 formal config 后，才可得到可比较的正式结果。

## 4. GPU Smoke Study

按原样运行仅使用 validation 选择的 smoke study：

```bat
python experiments/run_p12_comparison.py study --config configs/mnist_p12_gpu_smoke.yaml --checkpoint checkpoints/mnist_dense_baseline.pth
```

该 profile 设计为小规模、Windows 稳定验证：`cuda:0`、FP16 AMP、deterministic、无 DataLoader worker、一次 search iteration、每轮一个 candidate、零 recovery epoch，并且固定执行以下六种方法：

```text
dense_baseline
dense_small
oneshot_magnitude
oneshot_wanda
iterative_structured_level1
autonomous_search
```

保存完整控制台输出，特别是命令打印的准确 study 目录。预期输出根目录为：

```text
results/p12_comparison_gpu_smoke/
```

如实验预计超过五分钟，启动后立即记录并通知：

```text
实验名称: GPU P1.2 smoke study
预计时长: <根据本机实测速度填写>
输出目录: <命令实际打印的 study 目录>
```

### Smoke 冻结 Gate

运行 `report-test` 前，检查 `<SMOKE_STUDY_DIR>\manifest.json`，必须确认：

- `protocol` 是 validation-only 的 P1.2 study protocol。
- `selection_frozen` 为 `true`。
- comparison records 恰好六条。
- 每个选中结果都有 checkpoint 和记录的 SHA-256。
- source checkpoint hash 与批准的基线 checkpoint digest 相同。
- runtime block 指向 CUDA，并包含 CUDA/cuDNN metadata。
- manifest 包含预期 config hash、Git SHA 和 split metadata。
- `<SMOKE_STUDY_DIR>\final_test_report.json` 尚不存在。

预期 study 文件包括：

```text
resolved_config.yaml
resolved_config.json
comparison.json
comparison.csv
manifest.json
summary.json
<六种方法各一个 checkpoint>
```

任一条件不满足则停止。不得通过执行 `report-test` 来诊断或绕过冻结失败。

## 5. 冻结后的 Smoke Test Report

使用 smoke 命令实际打印出的目录：

```bat
python experiments/run_p12_comparison.py report-test --config configs/mnist_p12_gpu_smoke.yaml --checkpoint checkpoints/mnist_dense_baseline.pth --study-dir <SMOKE_STUDY_DIR>
```

这是本交接流程中首次允许加载官方 MNIST test 数据的命令。runner 会在测试前检查 config hash、source checkpoint hash、选中 checkpoint 文件及其 hash、冻结状态和 split metadata。

仅当 `<SMOKE_STUDY_DIR>\final_test_report.json` 存在，且其中有六个方法、只引用 frozen smoke checkpoints 时，本阶段才通过。manifest 与 final report 必须一并保留。

## 6. 正式 GPU Study

仅当 smoke study 和 smoke frozen test report 均通过后，才能启动正式实验：

```bat
python experiments/run_p12_comparison.py study --config configs/mnist_p12_gpu_study.yaml --checkpoint checkpoints/mnist_dense_baseline.pth
```

正式实验使用独立输出根目录：

```text
results/p12_comparison_gpu/
```

它必须与 CPU artifacts 和 smoke artifacts 分离。正式 profile 特征如下：

- `cuda:0`、BF16 AMP、mixed precision enabled。
- deterministic disabled、cuDNN benchmark enabled、TF32 enabled。
- 两个 worker、persistent workers 和 prefetch factor 2。
- 三次 search iteration、每轮两个 candidate、两次 recovery epoch。
- 与 smoke 相同的六种 comparison methods。

记录命令实际打印的 `<FORMAL_STUDY_DIR>`。不得把正式 report 指向 smoke 目录。

如正式实验预计超过五分钟，启动后立即记录并通知：

```text
实验名称: GPU P1.2 formal study
预计时长: <根据本机实测速度填写>
输出目录: <命令实际打印的 formal study 目录>
```

### 正式实验冻结 Gate

检查 `<FORMAL_STUDY_DIR>\manifest.json`，按照与 smoke 相同的标准确认：

- `selection_frozen: true`。
- 恰好六条 records。
- CUDA runtime 和 policy metadata 存在。
- source checkpoint hash 与批准值一致。
- config hash、Git SHA 与 train/validation split metadata 存在。
- 六个选中 checkpoint 文件均存在且 hash 匹配。
- 运行 report-test 前不存在 `final_test_report.json`。

## 7. 冻结后的正式 Test Report

使用正式实验实际生成的目录：

```bat
python experiments/run_p12_comparison.py report-test --config configs/mnist_p12_gpu_study.yaml --checkpoint checkpoints/mnist_dense_baseline.pth --study-dir <FORMAL_STUDY_DIR>
```

通过标准为 `<FORMAL_STUDY_DIR>\final_test_report.json` 存在，含六个方法，并且仅引用对应 formal manifest 中冻结的 checkpoint。任何 hash、config、selection 或 split mismatch 都是预期的安全失败，不得绕过或手工编辑 manifest。

## 8. 必须回传的证据包

两个 study 目录在审查完成前必须保留。回传以下内容或提供批准的安全传输位置：

1. 已填写的交接身份信息。
2. 完整 `nvidia-smi` 输出，以及 Python/Torch runtime version 输出。
3. 完整成功的 FP16 自检输出，以及 BF16 自检输出（如已执行）。
4. 准确的 smoke/formal study 命令与命令打印的目录。
5. 对每个 study 目录，回传：
   - `manifest.json`。
   - `resolved_config.yaml` 和/或 `resolved_config.json`。
   - `comparison.json` 与 `comparison.csv`。
   - `summary.json`。
   - `final_test_report.json`。
   - 六个选中 checkpoint，或其 SHA-256 与批准的安全存储位置。
6. 异常日志，列明 warnings、重试、配置变动；正常情况下应明确记录为无。

不得将证据包、实验结果、数据集、缓存或 checkpoint 提交到 Git。研究负责人确认 manifest、hash、split metadata 和报告可读且一致前，输出目录必须保持不变。

## 9. 故障处理与升级

| 现象 | 必须采取的动作 | 是否可继续 |
|---|---|---|
| `nvidia-smi` 失败或未显示 GPU | 修复驱动、调度器分配、容器 GPU passthrough 或远程会话，然后重跑第 1 节。 | 否 |
| PyTorch 显示 CUDA unavailable | 安装与驱动兼容的官方 CUDA wheel，重跑第 2-3 节。 | 否 |
| `cuda:0` 不可用但有其他 GPU | 获取批准的、显式版本化的 device/config 变更；不得本地编辑 YAML。 | 否 |
| CUDA out-of-memory | 保存错误与显存状态；申请批准的更小 batch size 配置。不得留下未记录的本地修改。 | 否 |
| Windows DataLoader worker failure | Smoke 必须保持 `num_workers: 0`；正式实验须回传错误并申请 profile 修订。 | Smoke 可继续；正式实验阻塞 |
| BF16 unsupported | 回传 GPU capability 和 BF16 错误；仅在获得批准的新 formal config 后执行。 | 正式实验不可继续 |
| MNIST 下载或网络失败 | 恢复网络，或提供批准的数据缓存，然后重跑被中断阶段。 | 恢复前不可继续 |
| `report-test` 出现 config/checkpoint/split hash mismatch | 该 study 对 test reporting 无效。恢复批准输入后重新执行 `study`。 | 否 |
| 输出目录冲突或旧目录误用 | 仅使用当前命令新打印的匹配目录；不得复用旧 run。 | 可继续，前提是目录匹配 |

## 10. 最终完成清单

仅在对应证据已存在时勾选：

```text
[ ] 已记录批准的 Git SHA 和 checkpoint SHA
[ ] 已保存 nvidia-smi 输出
[ ] 已保存 CUDA-enabled PyTorch 版本信息
[ ] FP16 check_gpu.py 已通过
[ ] BF16 check_gpu.py 已通过，或已升级并获得批准的替代配置
[ ] Smoke study 已完成且 selection_frozen=true
[ ] Smoke manifest 和六条 records 已核验
[ ] Smoke frozen report-test 已完成
[ ] Formal study 已完成且 selection_frozen=true
[ ] Formal manifest 和六条 records 已核验
[ ] Formal frozen report-test 已完成
[ ] 两个结果目录均已保留
[ ] 证据包已回传
[ ] 未提交任何生成物
```

GPU 结果只有在研究负责人根据批准的源 revision 与 checkpoint 核验回传的 manifest、checkpoint hashes、split metadata、runtime metadata 和 frozen reports 后，才可作为研究证据使用。
