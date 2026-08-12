# 项目工作流程说明

## 📋 开发环境配置

### 本地环境（当前）
- **硬件**: CPU only (无 CUDA)
- **用途**: 代码开发、逻辑实现、小规模测试
- **AI 助手**: Claude Code 在本地运行
- **适合任务**: ✅ 标记的所有任务

### 云服务器环境
- **平台**: 英博云
- **配置**: GPU-4090D (24GB VRAM) + 10core CPU + 100GB 内存
- **用途**: 大规模训练、论文实验复现
- **适合任务**: ⚠️ 和 🔥 标记的任务

---

## 🔄 工作流程

### 1. 本地开发（主要工作区）
```bash
# 在本地使用 Claude Code 进行开发
# 工作目录: d:\Compression\Model-Compression

# 任务类型:
# ✅ 代码实现（算法逻辑、工具函数、控制器）
# ✅ 单元测试编写
# ✅ MNIST 小规模验证
# ✅ 文档编写和项目管理
```

### 2. 提交到 Git
```bash
# 本地完成开发后，提交到 GitHub
git add .
git commit -m "描述你的改动"
git push origin main
```

### 3. 云服务器同步
```bash
# SSH 连接到云服务器
ssh root@<服务器IP> -p <端口>

# 首次克隆项目
git clone https://github.com/upUpUP-cody/Model-Compression.git
cd Model-Compression

# 后续同步（拉取本地的更新）
git pull origin main
```

### 4. 云服务器执行高配置任务
```bash
# 在云服务器上运行需要 GPU 的任务
python experiments/train_cifar10.py
python experiments/squad_experiment.py

# 运行完成后，结果会保存到 results/ 目录
```

### 5. 同步结果回本地
```bash
# 云服务器上提交结果
git add results/
git commit -m "Add experiment results"
git push origin main

# 本地拉取结果
git pull origin main
```

---

## ⚠️ 重要注意事项

### 文件管理
- ✅ **代码文件** → Git 同步
- ✅ **配置文件** → Git 同步
- ✅ **脚本文件** → Git 同步
- ✅ **实验结果**（JSON/CSV/图表）→ Git 同步
- ❌ **大型数据集**（CIFAR-10/SQuAD）→ **不同步**，云服务器单独下载
- ❌ **训练好的模型权重**（.pth/.pt）→ **不同步**，使用云存储或按需下载
- ❌ **临时文件/缓存** → 已在 .gitignore 中排除

### Git 配置建议
```bash
# 在云服务器上配置 Git 用户信息
git config --global user.name "upUpUP-cody"
git config --global user.email "your-email@example.com"

# 配置 SSH key（推荐）或使用 HTTPS + token
```

### 数据集处理策略
```bash
# 本地：仅下载 MNIST（小，自动下载）
# 云服务器：下载 CIFAR-10 和 SQuAD

# 云服务器数据准备脚本
python scripts/download_datasets.py --dataset cifar10
python scripts/download_datasets.py --dataset squad
```

---

## 📊 任务分配参考

### 本地完成（✅ CPU 可运行）
- [x] 阶段 0: 环境搭建
- [ ] 步骤 A.1: MNIST + MLP 模型实现
- [ ] 步骤 A.2-A.3: 剪枝算子实现
- [ ] 阶段 B: 前沿分析算法实现
- [ ] 阶段 C: 恢复策略实现（算法逻辑）
- [ ] 阶段 D: 控制器实现
- [ ] 步骤 E.1-E.2: 集成与单元测试
- [ ] 步骤 F.1: MNIST 端到端测试

### 云服务器完成（⚠️🔥 需要 GPU）
- [ ] 步骤 A.1: CIFAR-10 / SQuAD 模型训练
- [ ] 步骤 C.3-C.4: LoRA 恢复与知识蒸馏训练
- [ ] 步骤 F.2: CIFAR-10 完整流程测试
- [ ] 步骤 F.3-F.5: 论文实验复现（SQuAD + Qwen-0.5B）
- [ ] 阶段 G: 扩展实验

---

## 🤖 Claude Code GPU 任务提醒机制

### 任务分类与提醒规则

Claude Code 会在执行任务前自动识别硬件需求并给出提醒：

#### ✅ **自动继续（本地 CPU）**
不会给出提醒，直接在本地执行：
- MNIST 相关的所有开发和训练
- 所有算法实现（剪枝、LoRA、知识蒸馏逻辑）
- 单元测试和集成测试
- 代码重构、文档编写
- 小规模数据处理和验证

#### ⚠️ **给出建议（建议 GPU）**
会提示但允许本地继续：
- CIFAR-10 训练（CPU 慢 5-10 倍）
- 中等规模模型训练
- 提示示例: 
  > ⚠️ 此任务建议使用 GPU (预计 CPU 需 2 小时，GPU 仅需 15 分钟)
  > 
  > 选项：
  > 1. 继续本地 CPU 训练（会较慢）
  > 2. 切换到云服务器执行

#### 🔥 **必须停止（必须 GPU）**
强制要求切换到服务器：
- SQuAD + Qwen-0.5B 训练
- 论文实验复现（步骤 F.3-F.5）
- 大模型相关任务
- 提示示例:
  > 🔥 此任务必须使用 GPU (CPU 不可行，预计需要数天且可能内存溢出)
  > 
  > 需要执行的操作：
  > 1. 提交当前本地代码到 Git
  > 2. 连接到云服务器
  > 3. 在服务器上执行: `git pull && python experiments/train_squad.py`

### 任务清单标注

所有任务已在 PROJECT_PLAN.md 中标注硬件需求：

```
✅ [CPU 可运行]    - 本地直接执行
⚠️ [建议 GPU]     - 给出提醒，允许本地继续
🔥 [必须 GPU]     - 强制要求切换服务器
```

### 云服务器切换流程

当遇到 🔥 任务时，Claude Code 会引导你完成以下步骤：

```bash
# 1. 本地提交代码
git add .
git commit -m "准备云端实验"
git push

# 2. 连接云服务器
ssh root@<IP> -p <PORT>

# 3. 同步代码
cd Model-Compression
git pull

# 4. 执行任务
python experiments/train_squad.py --config configs/squad_config.yaml

# 5. 推送结果
git add results/
git commit -m "Add experiment results"
git push

# 6. 本地拉取
exit  # 退出服务器
git pull
```

---

## 🛠️ Claude Code 使用说明

### 在本地
- Claude Code **在本地运行**，可以直接读写本地文件
- 使用 Claude Code 进行代码开发、调试、重构
- Claude Code 会通过 Git 命令同步代码

### 在云服务器
- **不需要** 在云服务器上安装 Claude Code
- 云服务器仅用于执行训练脚本
- 通过 SSH 手动执行命令，或编写自动化脚本

### 协作模式
```
本地 (Claude Code)        Git (GitHub)         云服务器 (纯执行)
    │                         │                      │
    │ 1. 编写代码             │                      │
    │ ──────────────────────> │                      │
    │ (git push)              │                      │
    │                         │ 2. 拉取代码          │
    │                         │ <────────────────────│
    │                         │ (git pull)           │
    │                         │                      │
    │                         │ 3. 执行训练          │
    │                         │ ────────────────────>│
    │                         │ (python train.py)    │
    │                         │                      │
    │                         │ 4. 推送结果          │
    │                         │ <────────────────────│
    │ 5. 拉取结果             │ (git push)           │
    │ <────────────────────── │                      │
    │ (git pull)              │                      │
```

---

## 🔐 安全建议

### SSH 密钥配置
```bash
# 本地生成 SSH 密钥
ssh-keygen -t ed25519 -C "your-email@example.com"

# 添加公钥到云服务器
cat ~/.ssh/id_ed25519.pub
# 复制内容到云服务器的 ~/.ssh/authorized_keys
```

### 敏感信息管理
- ❌ **不要** 将 API keys、密码提交到 Git
- ✅ 使用环境变量或 `.env` 文件（已在 .gitignore）
- ✅ 云服务器配置单独存储在服务器本地

---

## 📝 快速命令参考

### 本地常用命令
```bash
# 查看项目状态
git status

# 提交改动
git add src/
git commit -m "Implement pruning operators"
git push

# 拉取云端结果
git pull
```

### 云服务器常用命令
```bash
# 连接服务器
ssh root@<IP> -p <PORT>

# 同步代码
cd Model-Compression
git pull

# 运行实验
python experiments/train_squad.py --gpu 0

# 查看 GPU 状态
nvidia-smi

# 后台运行（防止断开连接中断）
nohup python experiments/train_squad.py > output.log 2>&1 &

# 查看日志
tail -f output.log

# 提交结果
git add results/
git commit -m "Add SQuAD experiment results"
git push
```

---

## ✅ 当前进度检查点

- [x] 本地环境搭建完成
- [x] Git 仓库配置完成
- [ ] 云服务器实例创建
- [ ] 云服务器首次同步
- [ ] 完成 MNIST 本地验证
- [ ] 完成首次云端实验

---

## 📞 问题排查

### 本地遇到问题
- 使用 Claude Code 直接询问和调试
- 检查 Git 同步状态

### 云服务器遇到问题
- 检查 SSH 连接
- 查看运行日志
- 验证 GPU 可用性 (`nvidia-smi`)
- 确认依赖已安装

---

**最后更新**: 2026-08-12
**维护者**: upUpUP-cody
