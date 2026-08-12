"""
敏感度分析模块
实现多种层级敏感度评估方法，用于指导剪枝决策
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader


class SensitivityAnalyzer:
    """
    敏感度分析器
    评估每一层对模型性能的重要性
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Args:
            model: 待分析的模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)

    def compute_gradient_sensitivity(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        num_batches: int = 10
    ) -> Dict[str, float]:
        """
        计算基于梯度的敏感度
        敏感度 = 梯度范数的平均值

        Args:
            dataloader: 数据加载器
            criterion: 损失函数
            num_batches: 使用的批次数（用于加速）

        Returns:
            {layer_name: sensitivity_score}
        """
        self.model.train()
        sensitivity_scores = {}

        # 初始化累积梯度
        gradient_accumulator = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                gradient_accumulator[name] = 0.0

        # 累积多个批次的梯度
        batch_count = 0
        for data, target in dataloader:
            if batch_count >= num_batches:
                break

            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            self.model.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)

            # 反向传播
            loss.backward()

            # 累积梯度范数
            for name, param in self.model.named_parameters():
                if name in gradient_accumulator and param.grad is not None:
                    gradient_accumulator[name] += torch.norm(param.grad, p=2).item()

            batch_count += 1

        # 计算平均敏感度
        for name, grad_sum in gradient_accumulator.items():
            sensitivity_scores[name] = grad_sum / batch_count

        return sensitivity_scores

    def compute_weight_magnitude_sensitivity(self) -> Dict[str, float]:
        """
        计算基于权重幅度的敏感度
        敏感度 = 权重的 L2 范数

        Returns:
            {layer_name: sensitivity_score}
        """
        sensitivity_scores = {}

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                sensitivity_scores[name] = torch.norm(param.data, p=2).item()

        return sensitivity_scores

    def compute_wanda_sensitivity(
        self,
        dataloader: DataLoader,
        num_batches: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        计算 Wanda 分数 (Weight AND Activation)
        Wanda Score = |Weight| * |Activation|

        Args:
            dataloader: 数据加载器
            num_batches: 使用的批次数

        Returns:
            {layer_name: wanda_scores (神经元级别)}
        """
        self.model.eval()

        # 注册前向钩子来捕获激活值
        activations = {}
        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                if name not in activations:
                    activations[name] = []
                # 对于线性层，output shape 是 (batch, features)
                activations[name].append(output.detach())
            return hook

        # 为所有 Linear 层注册钩子
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(get_activation(name)))

        # 前向传播收集激活值
        batch_count = 0
        with torch.no_grad():
            for data, _ in dataloader:
                if batch_count >= num_batches:
                    break
                data = data.to(self.device)
                _ = self.model(data)
                batch_count += 1

        # 移除钩子
        for hook in hooks:
            hook.remove()

        # 计算 Wanda 分数
        wanda_scores = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in activations:
                # 权重: (out_features, in_features)
                weight = module.weight.data

                # 激活值: list of (batch, out_features)
                activation_list = activations[name]
                # 合并所有批次: (total_samples, out_features)
                all_activations = torch.cat(activation_list, dim=0)

                # 计算每个神经元的平均激活幅度
                avg_activation = torch.mean(torch.abs(all_activations), dim=0)  # (out_features,)

                # 计算每个神经元的权重幅度
                weight_magnitude = torch.norm(weight, p=2, dim=1)  # (out_features,)

                # Wanda 分数 = 权重幅度 * 激活幅度
                wanda = weight_magnitude * avg_activation

                wanda_scores[name] = wanda

        return wanda_scores

    def compute_layer_sensitivity(
        self,
        dataloader: DataLoader,
        method: str = 'wanda',
        num_batches: int = 10
    ) -> Dict[str, float]:
        """
        统一接口：计算层级敏感度

        Args:
            dataloader: 数据加载器
            method: 敏感度计算方法
                - 'gradient': 梯度敏感度
                - 'magnitude': 权重幅度
                - 'wanda': Wanda 分数
            num_batches: 使用的批次数

        Returns:
            {layer_name: sensitivity_score}
        """
        if method == 'gradient':
            criterion = nn.CrossEntropyLoss()
            return self.compute_gradient_sensitivity(dataloader, criterion, num_batches)

        elif method == 'magnitude':
            return self.compute_weight_magnitude_sensitivity()

        elif method == 'wanda':
            wanda_scores = self.compute_wanda_sensitivity(dataloader, num_batches)
            # 将神经元级别的分数聚合为层级分数
            layer_scores = {}
            for name, scores in wanda_scores.items():
                layer_scores[name] = torch.mean(scores).item()
            return layer_scores

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_neuron_importance(
        self,
        layer_name: str,
        dataloader: DataLoader,
        method: str = 'wanda',
        num_batches: int = 10
    ) -> torch.Tensor:
        """
        获取指定层中每个神经元的重要性分数

        Args:
            layer_name: 层名称
            dataloader: 数据加载器
            method: 重要性计算方法
            num_batches: 使用的批次数

        Returns:
            重要性分数张量 (out_features,)
        """
        if method == 'wanda':
            wanda_scores = self.compute_wanda_sensitivity(dataloader, num_batches)
            return wanda_scores.get(layer_name, None)

        elif method == 'magnitude':
            # 获取层
            module = self._get_module_by_name(layer_name)
            if isinstance(module, nn.Linear):
                return torch.norm(module.weight.data, p=2, dim=1)

        elif method == 'gradient':
            # 需要实现神经元级别的梯度敏感度
            raise NotImplementedError("Neuron-level gradient sensitivity not implemented")

        else:
            raise ValueError(f"Unknown method: {method}")

    def _get_module_by_name(self, name: str) -> nn.Module:
        """通过名称获取模块"""
        parts = name.split('.')
        module = self.model

        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        return module

    def rank_layers_by_sensitivity(
        self,
        dataloader: DataLoader,
        method: str = 'wanda',
        num_batches: int = 10
    ) -> List[Tuple[str, float]]:
        """
        按敏感度对层进行排序

        Args:
            dataloader: 数据加载器
            method: 敏感度计算方法
            num_batches: 使用的批次数

        Returns:
            排序后的 [(layer_name, sensitivity_score)] 列表
            按敏感度从高到低排序
        """
        sensitivity_scores = self.compute_layer_sensitivity(
            dataloader, method, num_batches
        )

        # 排序
        ranked = sorted(
            sensitivity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked


def compare_sensitivity_methods(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
    num_batches: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    比较不同敏感度计算方法的结果

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 计算设备
        num_batches: 使用的批次数

    Returns:
        {method_name: {layer_name: score}}
    """
    analyzer = SensitivityAnalyzer(model, device)

    results = {}

    # 权重幅度（最快）
    print("计算权重幅度敏感度...")
    results['magnitude'] = analyzer.compute_layer_sensitivity(
        dataloader, method='magnitude', num_batches=num_batches
    )

    # Wanda 分数（中等速度）
    print("计算 Wanda 敏感度...")
    results['wanda'] = analyzer.compute_layer_sensitivity(
        dataloader, method='wanda', num_batches=num_batches
    )

    # 梯度敏感度（较慢）
    print("计算梯度敏感度...")
    results['gradient'] = analyzer.compute_layer_sensitivity(
        dataloader, method='gradient', num_batches=num_batches
    )

    return results


if __name__ == '__main__':
    # 测试敏感度分析
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    print("=" * 70)
    print("测试敏感度分析")
    print("=" * 70)

    # 创建简单模型
    from src.models.dense_baseline import create_mnist_baseline
    from src.utils.data_loader import get_mnist_loaders

    print("\n加载模型和数据...")
    model = create_mnist_baseline()
    train_loader, _ = get_mnist_loaders(batch_size=64, num_workers=0)

    print("\n创建敏感度分析器...")
    analyzer = SensitivityAnalyzer(model, device='cpu')

    # 测试权重幅度
    print("\n[1] 权重幅度敏感度:")
    magnitude_scores = analyzer.compute_layer_sensitivity(
        train_loader, method='magnitude'
    )
    for name, score in list(magnitude_scores.items())[:5]:
        print(f"  {name}: {score:.4f}")

    # 测试 Wanda
    print("\n[2] Wanda 敏感度 (使用前5个批次):")
    wanda_scores = analyzer.compute_layer_sensitivity(
        train_loader, method='wanda', num_batches=5
    )
    for name, score in list(wanda_scores.items())[:5]:
        print(f"  {name}: {score:.4f}")

    # 排序
    print("\n[3] 按 Wanda 分数排序:")
    ranked = analyzer.rank_layers_by_sensitivity(
        train_loader, method='wanda', num_batches=5
    )
    for name, score in ranked[:5]:
        print(f"  {name}: {score:.4f}")

    print("\n[SUCCESS] 敏感度分析测试通过!")
