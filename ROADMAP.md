# 项目实施路线图 - 快速参考

## 📋 总体进度追踪

```
[⬜️] 阶段 0: 环境搭建 (3-5天)
[⬜️] 阶段 A: 基础组件 (7-10天)
[⬜️] 阶段 B: 前沿分析 (5-7天)
[⬜️] 阶段 C: 恢复策略 (7-10天)
[⬜️] 阶段 D: 压缩控制器 (5-7天)
[⬜️] 阶段 E: 端到端集成 (5-7天)
[⬜️] 阶段 F: 实验验证 (7-10天)
```

---

## 🚀 快速启动指南

### Day 1-2: 立即可做
```bash
# 1. 创建项目结构
mkdir -p src/{models,pruning,recovery,controller,evaluation,utils}
mkdir -p configs experiments tests data checkpoints logs results

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install torch torchvision transformers numpy pandas matplotlib seaborn pytest wandb

# 4. 下载 MNIST 数据集
python -c "from torchvision.datasets import MNIST; MNIST('./data', download=True)"

# 5. 运行第一个 baseline
# (需要先实现 src/models/dense_baseline.py)
```

### Week 1: 基础验证
- **目标**: 在 MNIST 上训练一个 3 层 MLP 达到 >98% 准确率
- **交付物**:
  - `src/models/mlp.py`
  - `experiments/train_mnist_baseline.py`
  - 训练日志和模型检查点

### Week 2-3: 核心功能
- **目标**: 实现剪枝 + 简单恢复
- **交付物**:
  - `src/pruning/structured_pruning.py`
  - `src/pruning/sensitivity.py`
  - `src/recovery/reconstruction.py`
  - 单元测试通过

### Week 4: 端到端
- **目标**: 完整的自主搜索循环可运行
- **交付物**:
  - `src/autonomous_search.py`
  - MNIST 实验结果
  - 稀疏度-准确率曲线图

---

## 📊 实验检查清单

### 实验 E0: 环境验证
```python
# ✅ 检查项
- [ ] PyTorch 安装成功
- [ ] GPU 可用 (如果有)
- [ ] MNIST 数据集加载成功
- [ ] 简单模型可训练
```

### 实验 E1: Dense Baseline (论文 Table 1)
```python
# 配置
model: MLP-3层 / ResNet-18 / BERT-base
dataset: MNIST / CIFAR-10 / SQuAD
metric: Accuracy / F1

# ✅ 检查项
- [ ] Dense 模型训练到收敛
- [ ] 记录准确率、参数量、推理时间
- [ ] 保存模型检查点
```

### 实验 E2: One-shot Pruning
```python
# 配置
pruning_method: Magnitude / Wanda / SparseGPT
target_sparsity: [0.3, 0.5, 0.7, 0.9, 0.95]

# ✅ 检查项
- [ ] 每个稀疏度下剪枝成功
- [ ] 记录剪枝后准确率 (无恢复)
- [ ] 绘制稀疏度 vs 准确率曲线
```

### 实验 E3: Recovery 对比
```python
# 配置
recovery_levels: [0, 1, 2, 3]  # None, Reconstruct, LoRA, Distillation
sparsity: 0.9

# ✅ 检查项
- [ ] Level 0: 直接评估
- [ ] Level 1: 重建训练 5 epochs
- [ ] Level 2: LoRA 恢复
- [ ] Level 3: 自蒸馏
- [ ] 对比各级别的准确率提升
```

### 实验 E4: 端到端自主搜索
```python
# 配置
max_iterations: 10
target_sparsity: 0.9
controller: heuristic

# ✅ 检查项
- [ ] 搜索循环完整运行
- [ ] 每轮记录 capability gap
- [ ] 最终模型达到目标稀疏度
- [ ] 准确率超过 one-shot baseline
```

### 实验 E5: 消融研究
```python
# 对比组
variants: [
    "no_sensitivity",      # 随机选择剪枝层
    "no_frontier",         # 不使用前沿分析
    "no_controller",       # 固定剪枝策略
    "no_recovery",         # 无恢复
    "full_method"          # 完整方法
]

# ✅ 检查项
- [ ] 每个变体独立实验
- [ ] 记录准确率和搜索轮数
- [ ] 统计显著性检验
```

---

## 🎯 里程碑时间线

```
Week 1  ████░░░░░░░░░░░░░░░░  Checkpoint 1: Dense Baseline ✓
Week 2  ████████░░░░░░░░░░░░  实现剪枝和敏感度分析
Week 3  ████████████░░░░░░░░  Checkpoint 2: One-shot Pruning ✓
Week 4  ████████████████░░░░  端到端集成
Week 5  ████████████████░░░░  Checkpoint 3: 完整搜索循环 ✓
Week 6  ████████████████████  实验验证和论文复现
```

---

## 🐛 常见问题排查

### 问题 1: 剪枝后模型无法运行
**症状**: `RuntimeError: size mismatch`
**原因**: 层间维度不匹配
**解决**:
```python
# 确保剪枝时同步更新相邻层
if prune_layer_i:
    layer_i.out_features = new_dim
    layer_i_plus_1.in_features = new_dim
```

### 问题 2: 恢复训练不收敛
**症状**: Loss 震荡或 NaN
**原因**: 学习率过大
**解决**:
```python
# 使用更小的学习率和 warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100)
```

### 问题 3: GPU 内存不足
**症状**: `CUDA out of memory`
**解决**:
```python
# 1. 减小 batch size
batch_size = 16  # 从 64 降低

# 2. 梯度累积
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. 混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    loss = model(batch)
```

### 问题 4: 搜索过程太慢
**症状**: 单轮迭代 > 1 小时
**优化**:
```python
# 1. 使用更小的 cheap critic 数据集
mini_dataset = random.sample(train_dataset, k=500)

# 2. 并行评估候选
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    scores = list(executor.map(cheap_critic, candidates))

# 3. 缓存敏感度计算
@functools.lru_cache(maxsize=32)
def compute_sensitivity(model_state_hash):
    ...
```

---

## 📝 每日开发日志模板

```markdown
## YYYY-MM-DD

### 今日目标
- [ ] 任务 1
- [ ] 任务 2

### 完成内容
- ✅ 实现了 XXX 功能
- ✅ 修复了 YYY bug
- 📊 实验结果: Accuracy = XX%

### 遇到的问题
1. **问题描述**: ...
   **解决方案**: ...

### 明日计划
- [ ] 任务 1
- [ ] 任务 2

### 代码变更
- 新增文件: `src/xxx.py`
- 修改文件: `src/yyy.py`
- Commit: `git commit -m "..."`
```

---

## 🔗 快速链接

- 详细计划: [PROJECT_PLAN.md](PROJECT_PLAN.md)
- 论文原文: `Autonomous Lottery Ticket Discovery (1).pdf`
- 实验细节: `Autonomous Lottery Ticket Discovery_experient.pdf`
- 代码仓库: https://github.com/upUpUP-cody/Model-Compression

---

## ✅ 下一步行动

**立即执行** (今天就可以开始):
1. ⬜️ 创建项目目录结构
2. ⬜️ 安装 Python 依赖
3. ⬜️ 下载 MNIST 数据集
4. ⬜️ 实现简单的 3 层 MLP
5. ⬜️ 训练 Dense Baseline

**本周目标**:
- ⬜️ 完成阶段 0 (环境搭建)
- ⬜️ 完成阶段 A.1 (Dense Baseline)
- ⬜️ 开始阶段 A.2 (结构化剪枝)

**需要决策**:
- [ ] 确认初始模型: MNIST+MLP 还是 CIFAR-10+ResNet?
- [ ] 确认 GPU 资源: 本地还是云端?
- [ ] 确认目标时间: MVP (4周) 还是完整版 (8周)?

---

**更新日期**: 2026-08-11
**当前阶段**: 阶段 0 - 环境搭建
**进度**: 0% (0/7 阶段完成)
