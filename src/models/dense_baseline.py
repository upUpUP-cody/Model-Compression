"""
Dense Baseline 模型实现
提供完整的密集神经网络作为性能基准
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from pathlib import Path


class MLP(nn.Module):
    """
    多层感知机 (MLP) 模型
    用于 MNIST 等简单任务
    """
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Args:
            input_dim: 输入维度 (MNIST: 28*28=784)
            hidden_dims: 隐藏层维度列表
            num_classes: 输出类别数
            dropout_rate: Dropout 比例
            use_batch_norm: 是否使用 Batch Normalization
        """
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm

        # 构建网络层
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # 线性层
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch Normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # 激活函数
            layers.append(nn.ReLU(inplace=True))

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # 特征提取器
        self.features = nn.Sequential(*layers)

        # 分类器
        self.classifier = nn.Linear(prev_dim, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, 1, 28, 28) 或 (batch_size, 784)

        Returns:
            输出 logits (batch_size, num_classes)
        """
        # 展平输入
        if x.dim() == 4:  # (batch_size, 1, 28, 28)
            x = x.view(x.size(0), -1)

        # 特征提取
        features = self.features(x)

        # 分类
        logits = self.classifier(features)

        return logits

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        获取模型参数量

        Args:
            trainable_only: 是否只统计可训练参数

        Returns:
            参数数量
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def get_model_size_mb(self) -> float:
        """
        获取模型大小 (MB)

        Returns:
            模型大小
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb


class ModelTrainer:
    """模型训练器"""
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            model: 待训练的模型
            device: 训练设备 ('cpu' 或 'cuda')
            learning_rate: 学习率
            weight_decay: L2 正则化系数
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def train_epoch(self, train_loader) -> tuple:
        """
        训练一个 epoch

        Args:
            train_loader: 训练数据加载器

        Returns:
            (平均损失, 准确率)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, test_loader) -> tuple:
        """
        评估模型

        Args:
            test_loader: 测试数据加载器

        Returns:
            (平均损失, 准确率)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)

            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader,
        test_loader,
        epochs: int = 20,
        verbose: bool = True,
        save_path: Optional[str] = None
    ) -> dict:
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            epochs: 训练轮数
            verbose: 是否打印训练信息
            save_path: 模型保存路径

        Returns:
            训练历史字典
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }

        best_acc = 0.0

        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 评估
            test_loss, test_acc = self.evaluate(test_loader)

            # 记录
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            # 打印
            if verbose:
                print(f"Epoch {epoch}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

            # 保存最佳模型
            if save_path and test_acc > best_acc:
                best_acc = test_acc
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'test_acc': test_acc,
                }, save_path)
                if verbose:
                    print(f"  [SAVED] Best model: {test_acc:.2f}%")

        return history


def create_mnist_baseline(
    hidden_dims: List[int] = [512, 256, 128],
    dropout_rate: float = 0.2
) -> MLP:
    """
    创建 MNIST Dense Baseline 模型

    Args:
        hidden_dims: 隐藏层维度
        dropout_rate: Dropout 比例

    Returns:
        MLP 模型实例
    """
    model = MLP(
        input_dim=784,
        hidden_dims=hidden_dims,
        num_classes=10,
        dropout_rate=dropout_rate,
        use_batch_norm=True
    )

    return model


if __name__ == '__main__':
    # 测试模型
    print("=" * 60)
    print("测试 MLP Dense Baseline 模型")
    print("=" * 60)

    # 创建模型
    model = create_mnist_baseline()

    # 打印模型信息
    print(f"\n模型架构:")
    print(model)

    print(f"\n模型统计:")
    print(f"  参数量: {model.get_num_parameters():,}")
    print(f"  模型大小: {model.get_model_size_mb():.2f} MB")

    # 测试前向传播
    dummy_input = torch.randn(4, 1, 28, 28)
    output = model(dummy_input)
    print(f"\n前向传播测试:")
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  输出样例: {output[0][:5]}")

    print("\n[SUCCESS] 模型测试通过!")


# 为了向后兼容，提供 DenseBaseline 别名
DenseBaseline = MLP
