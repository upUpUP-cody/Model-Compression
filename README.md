# Model Compression - Autonomous Lottery Ticket Discovery

基于《Autonomous Lottery Ticket Discovery》论文的神经网络自主压缩项目实现。

## 项目简介

本项目实现了一种**无需预训练父网络**的自主神经网络压缩方法，能够自动发现高性能的稀疏子网络（Lottery Tickets）。

### 核心特性

- 🎯 **自主搜索**: 无需迭代剪枝-重训练循环
- 🔍 **自我诊断**: 基于敏感度分析的智能剪枝
- 📊 **前沿分析**: 压缩-性能帕累托前沿追踪
- 🔄 **自适应恢复**: 多级恢复策略 (LoRA, Self-Distillation)
- 🤖 **智能控制器**: 决策式压缩流程管理

## 项目结构

```
Model-Compression/
├── src/                      # 源代码
│   ├── models/               # 神经网络模型定义
│   ├── pruning/              # 剪枝算法实现
│   ├── recovery/             # 恢复策略实现
│   ├── controller/           # 压缩控制器
│   ├── evaluation/           # 评估与指标
│   └── utils/                # 工具函数
├── configs/                  # 配置文件
├── experiments/              # 实验脚本
├── tests/                    # 单元测试
├── data/                     # 数据集目录
├── checkpoints/              # 模型检查点
├── logs/                     # 训练日志
├── results/                  # 实验结果
├── PROJECT_PLAN.md           # 详细实施计划
├── ROADMAP.md                # 快速参考路线图
└── README.md                 # 本文件
```

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/upUpUP-cody/Model-Compression.git
cd Model-Compression

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install torch torchvision transformers numpy pandas matplotlib seaborn pytest wandb
```

### 2. 数据集准备

```bash
# 下载 MNIST 数据集 (用于初期验证)
python -c "from torchvision.datasets import MNIST; MNIST('./data', download=True)"

# 下载 CIFAR-10 (可选)
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"
```

### 3. 运行示例

```bash
# 训练 Dense Baseline (待实现)
python experiments/train_mnist_baseline.py

# 运行自主压缩 (待实现)
python experiments/run_autonomous_search.py --config configs/mnist_mlp.yaml
```

## 实施计划

本项目采用**分阶段、模块化**的开发方式:

### 📅 开发阶段

| 阶段 | 内容 | 预计时间 | 状态 |
|------|------|----------|------|
| 阶段 0 | 环境搭建与基础设施 | 3-5 天 | ⬜️ 待开始 |
| 阶段 A | 基础组件实现 | 7-10 天 | ⬜️ 待开始 |
| 阶段 B | 前沿分析与候选生成 | 5-7 天 | ⬜️ 待开始 |
| 阶段 C | 恢复策略实现 | 7-10 天 | ⬜️ 待开始 |
| 阶段 D | 压缩控制器实现 | 5-7 天 | ⬜️ 待开始 |
| 阶段 E | 端到端流程集成 | 5-7 天 | ⬜️ 待开始 |
| 阶段 F | 实验验证 | 7-10 天 | ⬜️ 待开始 |

**详细计划**: 请查看 [PROJECT_PLAN.md](PROJECT_PLAN.md)  
**快速参考**: 请查看 [ROADMAP.md](ROADMAP.md)

### 🎯 里程碑

- **Checkpoint 1**: Dense Baseline 训练成功
- **Checkpoint 2**: One-shot Pruning + 简单恢复可运行
- **Checkpoint 3**: 端到端自主搜索循环可运行
- **Checkpoint 4**: MNIST 实验验证通过
- **Checkpoint 5**: 论文主实验复现

## 技术栈

- **深度学习框架**: PyTorch 2.0+
- **模型**: MLP, ResNet, BERT, Transformer
- **数据集**: MNIST, CIFAR-10, SQuAD 2.0, MMLU
- **剪枝方法**: Magnitude, Wanda, SparseGPT
- **恢复策略**: LoRA, Knowledge Distillation
- **实验跟踪**: Weights & Biases / TensorBoard

## 核心算法流程

```
1. Self-Diagnosis (自我诊断)
   └─> 计算层级敏感度 (Wanda/Gradient/Magnitude)

2. Frontier Profiling (前沿分析)
   └─> 评估 Capability Gap, Compression Gap, Context Gap

3. Ticket Proposal (候选生成)
   └─> 基于敏感度生成 K 个候选子网络

4. Cheap Critic (快速评估)
   └─> 使用小样本数据快速筛选

5. Controller Decision (控制器决策)
   └─> Accept / Reject / Rollback / Regrow

6. Recovery (恢复训练)
   └─> Level 0-3: None / Reconstruct / LoRA / Distillation

7. Evaluation (完整评估)
   └─> 在验证集上评估最终性能

8. Update & Iterate (更新迭代)
   └─> 更新历史，继续下一轮搜索
```

## 实验目标

### 最低标准 (MVP)
- ✅ Dense Baseline 在 MNIST 上 >98% 准确率
- ✅ One-shot Pruning 到 90% 稀疏度可执行
- ✅ 简单恢复策略能提升 5-10% 准确率
- ✅ 端到端搜索循环可运行

### 目标标准 (论文复现)
- 在 SQuAD 上达到论文报告的 F1 分数 (±2%)
- 稀疏度-准确率曲线与论文一致
- 优于 One-shot Wanda baseline 至少 10%
- 优于传统 IMP 至少 5%

## 论文参考

### 主要论文
1. **Autonomous Lottery Ticket Discovery** (本项目实现)
   - 自主彩票发现，无需预训练父网络

### 相关工作
2. **The Lottery Ticket Hypothesis** (Frankle & Carbin, ICLR 2019)
   - 提出彩票假设理论基础
3. **Wanda: Pruning by Weights and Activations** (Sun et al., 2023)
   - 权重×激活值的剪枝方法
4. **SparseGPT** (Frantar & Alistarh, ICML 2023)
   - 大语言模型的高效剪枝

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

### 开发流程
1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范
- 遵循 PEP 8 代码风格
- 添加必要的文档字符串
- 为新功能编写单元测试
- 运行 `pytest tests/` 确保测试通过

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- **作者**: upUpUP-cody
- **邮箱**: (待添加)
- **GitHub**: https://github.com/upUpUP-cody/Model-Compression
- **问题反馈**: https://github.com/upUpUP-cody/Model-Compression/issues

## 致谢

感谢以下开源项目的启发和支持:
- PyTorch Team
- Hugging Face Transformers
- Wanda Authors
- SparseGPT Authors

---

**项目状态**: 🚧 开发中  
**最后更新**: 2026-08-11  
**当前版本**: v0.1.0 (Pre-release)