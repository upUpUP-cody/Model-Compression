"""
重建恢复策略 (Level 1)
在剪枝后通过简单的微调训练来恢复模型性能
"""
import torch
import torch.nn as nn
from typing import Optional, Dict
from torch.utils.data import DataLoader


class ReconstructionRecovery:
    """
    Level 1: 重建恢复
    在压缩损伤后重新初始化并训练剪枝后的子网络
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            model: 剪枝后的模型
            device: 训练设备
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

    def train_epoch(self, train_loader: DataLoader) -> tuple:
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

        for data, target in train_loader:
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
    def evaluate(self, test_loader: DataLoader) -> tuple:
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

    def recover(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        执行重建恢复

        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            epochs: 恢复训练轮数
            verbose: 是否打印训练信息

        Returns:
            恢复历史字典
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }

        if verbose:
            print(f"\n (Level 1) - {epochs} ...")

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
                print(f"  Epoch {epoch}/{epochs}: "
                      f"Train Acc: {train_acc:.2f}%, "
                      f"Test Acc: {test_acc:.2f}%")

            # 记录最佳准确率
            if test_acc > best_acc:
                best_acc = test_acc

        if verbose:
            print(f"! : {best_acc:.2f}%")

        history['best_test_acc'] = best_acc

        return history


def quick_recovery(
    pruned_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    verbose: bool = True
) -> tuple:
    """
    快速恢复接口

    Args:
        pruned_model: 剪枝后的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        verbose: 是否打印信息

    Returns:
        (恢复后的模型, 历史记录)
    """
    recoverer = ReconstructionRecovery(
        model=pruned_model,
        device=device,
        learning_rate=learning_rate
    )

    history = recoverer.recover(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        verbose=verbose
    )

    return recoverer.model, history


if __name__ == '__main__':
    # 测试重建恢复
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    print("=" * 70)
    print(" (Level 1)")
    print("=" * 70)

    from src.models.dense_baseline import create_mnist_baseline
    from src.utils.data_loader import get_mnist_loaders
    from src.pruning.structured_pruning import StructuredPruning
    import copy

    # 加载数据
    print("\n...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128, num_workers=0)

    # 创建模型
    print("\n...")
    model = create_mnist_baseline()

    # 加载训练好的权重
    print("...")
    checkpoint = torch.load('./checkpoints/mnist_dense_baseline.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 评估原始模型
    print("\n...")
    recoverer = ReconstructionRecovery(model, device='cpu')
    _, original_acc = recoverer.evaluate(test_loader)
    print(f"  : {original_acc:.2f}%")

    # 剪枝模型 (50%)
    print("\n (50%)...")
    pruned_model = copy.deepcopy(model)
    pruner = StructuredPruning(pruned_model)

    layers_to_prune = [
        ('features.0', 'features.4'),
        ('features.4', 'features.8'),
        ('features.8', 'classifier'),
    ]

    for layer_name, next_layer_name in layers_to_prune:
        layer = pruner._get_layer_by_name(layer_name)
        importance = torch.norm(layer.weight.data, p=2, dim=1)
        keep_indices = pruner.prune_mlp_by_ratio(layer_name, 0.5, importance)
        pruner.prune_linear_block(layer_name, keep_indices, next_layer_name)

    # 评估剪枝后模型
    print("\n ()...")
    recoverer_pruned = ReconstructionRecovery(pruned_model, device='cpu')
    _, pruned_acc = recoverer_pruned.evaluate(test_loader)
    print(f"  : {pruned_acc:.2f}%")
    print(f"  : {original_acc - pruned_acc:.2f}%")

    # 恢复训练
    print("\n" + "=" * 70)
    history = recoverer_pruned.recover(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=5,
        verbose=True
    )

    # 最终结果
    print("\n" + "=" * 70)
    print(":")
    print(f"  : {original_acc:.2f}%")
    print(f"   (): {pruned_acc:.2f}%")
    print(f"  : {history['best_test_acc']:.2f}%")
    print(f"  : {history['best_test_acc'] - pruned_acc:.2f}%")

    print("\n[SUCCESS] !")
