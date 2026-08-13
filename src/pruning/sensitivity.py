"""
敏感度分析模块
实现多种层级敏感度评估方法，用于指导剪枝决策
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Sequence, Tuple
from torch.utils.data import DataLoader

from src.utils.device import resolve_device


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
        self.device = resolve_device(device)
        self.non_blocking = self.device.type == "cuda"
        self.model = model.to(self.device)

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

            data = data.to(self.device, non_blocking=self.non_blocking)
            target = target.to(self.device, non_blocking=self.non_blocking)

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

        if batch_count == 0:
            raise ValueError("dataloader must yield at least one batch")

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

    def compute_conv_wanda_importance(
        self,
        dataloader: DataLoader,
        layer_names: Sequence[str],
        num_batches: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-output-channel Wanda scores for named Conv2d layers."""
        if isinstance(num_batches, bool) or not isinstance(num_batches, int) or num_batches <= 0:
            raise ValueError("num_batches must be a positive integer")
        if not layer_names:
            raise ValueError("layer_names must not be empty")

        conv_layers: List[Tuple[str, nn.Conv2d]] = []
        for name in layer_names:
            module = self._get_module_by_name(name)
            if not isinstance(module, nn.Conv2d):
                raise ValueError(f"Layer {name} is not Conv2d")
            conv_layers.append((name, module))

        activation_sums: Dict[str, torch.Tensor] = {}
        activation_counts: Dict[str, int] = {}
        hooks = []
        was_training = self.model.training

        def get_activation(name: str):
            def hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                values = output.detach().abs()
                if values.ndim != 4:
                    raise ValueError(f"Conv layer {name} must produce four-dimensional activations")
                batch_sum = values.sum(dim=(0, 2, 3))
                if name not in activation_sums:
                    activation_sums[name] = batch_sum.clone()
                    activation_counts[name] = int(values.shape[0])
                else:
                    activation_sums[name].add_(batch_sum)
                    activation_counts[name] += int(values.shape[0])
            return hook

        self.model.eval()
        try:
            for name, module in conv_layers:
                hooks.append(module.register_forward_hook(get_activation(name)))

            batch_count = 0
            with torch.inference_mode():
                for data, _target in dataloader:
                    if batch_count >= num_batches:
                        break
                    self.model(data.to(self.device, non_blocking=self.non_blocking))
                    batch_count += 1

            if batch_count == 0:
                raise ValueError("dataloader must yield at least one batch")

            wanda_scores = {}
            for name, module in conv_layers:
                if name not in activation_sums or activation_counts[name] == 0:
                    raise ValueError(f"No activations collected for Conv layer {name}")
                average_activation = activation_sums[name] / activation_counts[name]
                weight_magnitude = torch.norm(module.weight.detach(), p=2, dim=(1, 2, 3))
                wanda_scores[name] = (weight_magnitude * average_activation).detach()
            return wanda_scores
        finally:
            for hook in hooks:
                hook.remove()
            self.model.train(was_training)

    def compute_wanda_importance(
        self,
        dataloader: DataLoader,
        num_batches: int = 10,
        layer_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute Wanda scores for hidden Linear layers or named Conv2d layers."""
        if layer_names is not None:
            return self.compute_conv_wanda_importance(dataloader, layer_names, num_batches)
        if isinstance(num_batches, bool) or not isinstance(num_batches, int) or num_batches <= 0:
            raise ValueError("num_batches must be a positive integer")

        linear_layers = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        hidden_layers = linear_layers[:-1]
        if not hidden_layers:
            raise ValueError("model must contain at least one hidden Linear layer")

        activation_sums: Dict[str, torch.Tensor] = {}
        activation_counts: Dict[str, int] = {}
        hooks = []
        was_training = self.model.training

        def get_activation(name: str):
            def hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                values = output.detach().abs()
                if values.ndim != 2:
                    raise ValueError(f"Linear layer {name} must produce two-dimensional activations")
                batch_sum = values.sum(dim=0)
                if name not in activation_sums:
                    activation_sums[name] = batch_sum.clone()
                    activation_counts[name] = int(values.shape[0])
                else:
                    activation_sums[name].add_(batch_sum)
                    activation_counts[name] += int(values.shape[0])
            return hook

        self.model.eval()
        try:
            for name, module in hidden_layers:
                hooks.append(module.register_forward_hook(get_activation(name)))

            batch_count = 0
            with torch.inference_mode():
                for data, _target in dataloader:
                    if batch_count >= num_batches:
                        break
                    self.model(data.to(self.device, non_blocking=self.non_blocking))
                    batch_count += 1

            if batch_count == 0:
                raise ValueError("dataloader must yield at least one batch")

            wanda_scores = {}
            for name, module in hidden_layers:
                if name not in activation_sums or activation_counts[name] == 0:
                    raise ValueError(f"No activations collected for hidden Linear layer {name}")
                average_activation = activation_sums[name] / activation_counts[name]
                weight_magnitude = torch.norm(module.weight.detach(), p=2, dim=1)
                wanda_scores[name] = (weight_magnitude * average_activation).detach()
            return wanda_scores
        finally:
            for hook in hooks:
                hook.remove()
            self.model.train(was_training)

    def compute_wanda_sensitivity(
        self,
        dataloader: DataLoader,
        num_batches: int = 10
    ) -> Dict[str, torch.Tensor]:
        """Backward-compatible alias for all hidden-Linear Wanda scores."""
        return self.compute_wanda_importance(dataloader, num_batches)

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
    print("...")
    results['magnitude'] = analyzer.compute_layer_sensitivity(
        dataloader, method='magnitude', num_batches=num_batches
    )

    # Wanda 分数（中等速度）
    print(" Wanda ...")
    results['wanda'] = analyzer.compute_layer_sensitivity(
        dataloader, method='wanda', num_batches=num_batches
    )

    # 梯度敏感度（较慢）
    print("...")
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
    print("")
    print("=" * 70)

    # 创建简单模型
    from src.models.dense_baseline import create_mnist_baseline
    from src.utils.data_loader import get_mnist_loaders

    print("\n...")
    model = create_mnist_baseline()
    train_loader, _ = get_mnist_loaders(batch_size=64, num_workers=0)

    print("\n...")
    analyzer = SensitivityAnalyzer(model, device='cpu')

    # 测试权重幅度
    print("\n[1] :")
    magnitude_scores = analyzer.compute_layer_sensitivity(
        train_loader, method='magnitude'
    )
    for name, score in list(magnitude_scores.items())[:5]:
        print(f"  {name}: {score:.4f}")

    # 测试 Wanda
    print("\n[2] Wanda  (5):")
    wanda_scores = analyzer.compute_layer_sensitivity(
        train_loader, method='wanda', num_batches=5
    )
    for name, score in list(wanda_scores.items())[:5]:
        print(f"  {name}: {score:.4f}")

    # 排序
    print("\n[3]  Wanda :")
    ranked = analyzer.rank_layers_by_sensitivity(
        train_loader, method='wanda', num_batches=5
    )
    for name, score in ranked[:5]:
        print(f"  {name}: {score:.4f}")

    print("\n[SUCCESS] !")
