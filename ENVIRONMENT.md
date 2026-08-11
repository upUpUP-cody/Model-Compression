# 环境配置信息

## Python 环境
- **Python 版本**: 3.11.9
- **pip 版本**: 24.0

## 核心依赖

### 深度学习框架
- PyTorch: 2.13.0 (CPU)
- TorchVision: 0.28.0
- TorchAudio: 2.11.0

### Transformer 生态
- Transformers: 5.15.0
- Datasets: 5.0.1
- Tokenizers: 0.22.2
- Accelerate: 1.14.0
- BitsAndBytes: 0.50.0

### 数据处理
- NumPy: 2.4.6
- Pandas: 3.0.5
- SciPy: 1.17.1

### 可视化
- Matplotlib: 3.11.1
- Seaborn: 0.13.2
- Plotly: 6.9.0

### 实验管理
- WandB: 0.28.1
- TensorBoard: 2.21.0

### 配置管理
- PyYAML: 6.0.3
- OmegaConf: 2.3.1

### 开发工具
- pytest: 9.1.1
- pytest-cov: 7.1.0
- Black: 26.5.1
- Flake8: 7.3.0
- isort: 8.0.1

## 硬件环境
- **CUDA**: 不可用
- **训练设备**: CPU only

## 验证脚本
1. `check_environment.py` - 完整环境验证
2. `test_imports.py` - 快速导入测试
3. `check_gpu.py` - GPU/CUDA 检测

## 安装说明
```bash
# 安装所有依赖
pip install -r requirements.txt

# 验证环境
python check_environment.py
```

---
*最后更新: 2026-08-11*
