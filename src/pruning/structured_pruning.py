"""
结构化剪枝实现
支持对 MLP 层、卷积层、注意力头进行结构化剪枝
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import copy


class StructuredPruning:
    """
    结构化剪枝器
    支持神经元级别、通道级别、注意力头级别的剪枝
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: 待剪枝的模型
        """
        self.model = model
        self.pruning_masks = {}

    def prune_mlp_neurons(
        self,
        layer_name: str,
        neuron_indices: List[int]
    ) -> None:
        """
        剪枝 MLP 层的指定神经元

        Args:
            layer_name: 层名称 (如 'features.0' 表示第一个线性层)
            neuron_indices: 要保留的神经元索引列表
        """
        # 获取目标层
        layer = self._get_layer_by_name(layer_name)

        if not isinstance(layer, nn.Linear):
            raise ValueError(f"Layer {layer_name} is not a Linear layer")

        # 剪枝权重和偏置
        with torch.no_grad():
            # 输出维度剪枝
            layer.weight.data = layer.weight.data[neuron_indices, :]
            if layer.bias is not None:
                layer.bias.data = layer.bias.data[neuron_indices]

            # 更新层的输出维度
            layer.out_features = len(neuron_indices)

        # 记录剪枝掩码
        self.pruning_masks[layer_name] = {
            'type': 'neuron',
            'indices': neuron_indices,
            'original_size': layer.weight.shape[0]
        }

    def prune_mlp_by_ratio(
        self,
        layer_name: str,
        prune_ratio: float,
        importance_scores: Optional[torch.Tensor] = None
    ) -> List[int]:
        """
        按比例剪枝 MLP 层

        Args:
            layer_name: 层名称
            prune_ratio: 剪枝比例 (0.0-1.0)
            importance_scores: 神经元重要性分数 (如果为None则使用权重L2范数)

        Returns:
            保留的神经元索引列表
        """
        layer = self._get_layer_by_name(layer_name)

        if not isinstance(layer, nn.Linear):
            raise ValueError(f"Layer {layer_name} is not a Linear layer")

        num_neurons = layer.out_features
        num_keep = int(num_neurons * (1 - prune_ratio))

        # 计算重要性分数
        if importance_scores is None:
            # 使用权重L2范数作为重要性
            importance_scores = torch.norm(layer.weight.data, p=2, dim=1)

        # 选择保留的神经元
        _, sorted_indices = torch.sort(importance_scores, descending=True)
        keep_indices = sorted(sorted_indices[:num_keep].tolist())

        return keep_indices

    def prune_linear_block(
        self,
        linear_layer_name: str,
        keep_indices: List[int],
        next_linear_layer_name: Optional[str] = None
    ) -> None:
        """
        剪枝线性层及其相关的 BatchNorm、Dropout 层

        Args:
            linear_layer_name: 线性层名称 (如 'features.0')
            keep_indices: 保留的神经元索引
            next_linear_layer_name: 下一个线性层名称 (用于调整输入维度)
        """
        # 剪枝当前线性层的输出
        linear_layer = self._get_layer_by_name(linear_layer_name)
        if isinstance(linear_layer, nn.Linear):
            with torch.no_grad():
                linear_layer.weight.data = linear_layer.weight.data[keep_indices, :]
                if linear_layer.bias is not None:
                    linear_layer.bias.data = linear_layer.bias.data[keep_indices]
                linear_layer.out_features = len(keep_indices)

        # 剪枝后续的 BatchNorm 层
        # 假设结构是 Linear(N) -> BatchNorm(N+1) -> ReLU(N+2) -> Dropout(N+3)
        parts = linear_layer_name.split('.')
        if len(parts) == 2 and parts[0] == 'features':
            linear_idx = int(parts[1])
            bn_idx = linear_idx + 1

            try:
                bn_layer = self._get_layer_by_name(f'features.{bn_idx}')
                if isinstance(bn_layer, nn.BatchNorm1d):
                    with torch.no_grad():
                        bn_layer.weight.data = bn_layer.weight.data[keep_indices]
                        bn_layer.bias.data = bn_layer.bias.data[keep_indices]
                        bn_layer.running_mean = bn_layer.running_mean[keep_indices]
                        bn_layer.running_var = bn_layer.running_var[keep_indices]
                        bn_layer.num_features = len(keep_indices)
            except:
                pass  # 没有 BatchNorm 层

        # 剪枝下一个线性层的输入
        if next_linear_layer_name:
            next_layer = self._get_layer_by_name(next_linear_layer_name)
            if isinstance(next_layer, nn.Linear):
                with torch.no_grad():
                    next_layer.weight.data = next_layer.weight.data[:, keep_indices]
                    next_layer.in_features = len(keep_indices)

    def prune_connected_layers(
        self,
        current_layer: str,
        next_layer: str,
        keep_indices: List[int]
    ) -> None:
        """
        剪枝两个连接的层（为了向后兼容保留此方法）

        Args:
            current_layer: 当前层名称
            next_layer: 下一层名称
            keep_indices: 保留的神经元索引
        """
        self.prune_linear_block(current_layer, keep_indices, next_layer)

    def prune_uniform(self, prune_ratio: float) -> int:
        """
        对所有隐藏层应用统一的剪枝比例

        注意：这个方法会原地修改模型

        Args:
            prune_ratio: 剪枝比例 (0.0-1.0)

        Returns:
            剪枝后的总参数量
        """
        # 找到所有线性层（按定义顺序）
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(name)

        if len(linear_layers) < 2:
            # 只有一个线性层（可能只是分类器），不剪枝
            return sum(p.numel() for p in self.model.parameters())

        # 从第一个隐藏层开始，到倒数第二个（不剪枝分类层）
        # 从后往前处理，避免影响前面层的索引计算
        for i in range(len(linear_layers) - 2, -1, -1):
            layer_name = linear_layers[i]
            next_layer_name = linear_layers[i + 1]

            # 计算保留的神经元索引
            layer = self._get_layer_by_name(layer_name)
            if not isinstance(layer, nn.Linear):
                continue

            num_neurons = layer.out_features
            num_keep = int(num_neurons * (1 - prune_ratio))

            if num_keep <= 0:
                num_keep = 1  # 至少保留一个神经元

            # 使用权重L2范数作为重要性
            importance_scores = torch.norm(layer.weight.data, p=2, dim=1)
            _, sorted_indices = torch.sort(importance_scores, descending=True)
            keep_indices = sorted(sorted_indices[:num_keep].tolist())

            # 应用剪枝（这会修改当前层的输出和下一层的输入）
            self.prune_linear_block(layer_name, keep_indices, next_layer_name)

        # 返回剪枝后的参数量
        return sum(p.numel() for p in self.model.parameters())

    def prune_by_layer(self, layer_ratios: Dict[str, float]) -> int:
        """
        按照指定的每层剪枝比例进行剪枝

        注意：这个方法会原地修改模型

        Args:
            layer_ratios: 每层的剪枝比例字典 {layer_name: prune_ratio}

        Returns:
            剪枝后的总参数量
        """
        # 找到所有线性层（按定义顺序）
        all_linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                all_linear_layers.append(name)

        # 过滤出需要剪枝的层（在 layer_ratios 中指定的）
        layers_to_prune = [name for name in all_linear_layers if name in layer_ratios]

        if not layers_to_prune:
            return sum(p.numel() for p in self.model.parameters())

        # 从后往前处理，避免影响前面层的索引计算
        for layer_name in reversed(layers_to_prune):
            prune_ratio = layer_ratios[layer_name]

            # 找到下一层
            layer_idx = all_linear_layers.index(layer_name)
            next_layer_name = all_linear_layers[layer_idx + 1] if layer_idx + 1 < len(all_linear_layers) else None

            # 计算保留的神经元索引
            layer = self._get_layer_by_name(layer_name)
            if not isinstance(layer, nn.Linear):
                continue

            # 如果是最后一层（分类器），跳过输出剪枝，但需要调整输入
            if next_layer_name is None:
                # 分类层不剪枝，跳过
                continue

            num_neurons = layer.out_features
            num_keep = int(num_neurons * (1 - prune_ratio))

            if num_keep <= 0:
                num_keep = 1  # 至少保留一个神经元

            # 使用权重L2范数作为重要性
            importance_scores = torch.norm(layer.weight.data, p=2, dim=1)
            _, sorted_indices = torch.sort(importance_scores, descending=True)
            keep_indices = sorted(sorted_indices[:num_keep].tolist())

            # 应用剪枝：修改当前层的输出
            layer.weight.data = layer.weight.data[keep_indices, :]
            if layer.bias is not None:
                layer.bias.data = layer.bias.data[keep_indices]
            layer.out_features = len(keep_indices)

            # 查找并调整紧跟的 BatchNorm 层（如果存在）
            # 在模型结构中，BatchNorm 通常紧跟 Linear 层
            parent_name = '.'.join(layer_name.split('.')[:-1])
            layer_idx_in_parent = int(layer_name.split('.')[-1])
            bn_name = f"{parent_name}.{layer_idx_in_parent + 1}"

            try:
                bn_layer = self._get_layer_by_name(bn_name)
                if isinstance(bn_layer, nn.BatchNorm1d):
                    # 调整 BatchNorm 的所有参数
                    bn_layer.weight.data = bn_layer.weight.data[keep_indices]
                    bn_layer.bias.data = bn_layer.bias.data[keep_indices]
                    bn_layer.running_mean = bn_layer.running_mean[keep_indices]
                    bn_layer.running_var = bn_layer.running_var[keep_indices]
                    bn_layer.num_features = len(keep_indices)
            except (AttributeError, ValueError):
                # 没有找到 BatchNorm 层，跳过
                pass

            # 修改下一个 Linear 层的输入维度
            next_layer = self._get_layer_by_name(next_layer_name)
            if isinstance(next_layer, nn.Linear):
                next_layer.weight.data = next_layer.weight.data[:, keep_indices]
                next_layer.in_features = len(keep_indices)

        # 返回剪枝后的参数量
        return sum(p.numel() for p in self.model.parameters())

    def create_pruned_model(
        self,
        pruning_config: Dict[str, float]
    ) -> nn.Module:
        """
        根据剪枝配置创建新的剪枝模型

        Args:
            pruning_config: 剪枝配置字典 {layer_name: prune_ratio}

        Returns:
            剪枝后的新模型
        """
        # 深拷贝原模型
        pruned_model = copy.deepcopy(self.model)
        pruner = StructuredPruning(pruned_model)

        # 按层剪枝
        for layer_name, prune_ratio in pruning_config.items():
            keep_indices = pruner.prune_mlp_by_ratio(layer_name, prune_ratio)

            # 找到下一层并一起剪枝
            next_layer = pruner._find_next_layer(layer_name)
            if next_layer:
                pruner.prune_connected_layers(layer_name, next_layer, keep_indices)

        return pruned_model

    def _get_layer_by_name(self, layer_name: str) -> nn.Module:
        """
        通过名称获取层

        Args:
            layer_name: 层名称 (支持点号分隔，如 'features.0')

        Returns:
            对应的层对象
        """
        parts = layer_name.split('.')
        module = self.model

        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        return module

    def _find_next_layer(self, layer_name: str) -> Optional[str]:
        """
        找到当前层的下一层

        Args:
            layer_name: 当前层名称

        Returns:
            下一层名称，如果不存在则返回 None
        """
        # 简单实现：假设是 features.N 格式
        if 'features' in layer_name:
            parts = layer_name.split('.')
            if len(parts) == 2 and parts[1].isdigit():
                next_idx = int(parts[1]) + 1
                # 跳过 BatchNorm 和 激活层，找到下一个 Linear
                for i in range(next_idx, next_idx + 10):
                    try:
                        next_layer = self._get_layer_by_name(f'features.{i}')
                        if isinstance(next_layer, nn.Linear):
                            return f'features.{i}'
                    except:
                        continue

        return None

    def get_sparsity(self) -> float:
        """
        计算模型的稀疏度

        Returns:
            稀疏度 (0.0-1.0)
        """
        total_params = 0
        zero_params = 0

        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0

    def get_model_info(self) -> Dict:
        """
        获取模型信息

        Returns:
            包含参数量、稀疏度等信息的字典
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'sparsity': self.get_sparsity(),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)
        }


def apply_pruning_mask(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """
    应用剪枝掩码到模型

    Args:
        model: 待剪枝的模型
        masks: 剪枝掩码字典 {layer_name: mask_tensor}
    """
    for name, param in model.named_parameters():
        if name in masks:
            with torch.no_grad():
                param.data *= masks[name]


def compute_layer_importance(
    layer: nn.Linear,
    method: str = 'l2'
) -> torch.Tensor:
    """
    计算层中每个神经元的重要性分数

    Args:
        layer: 线性层
        method: 计算方法 ('l2', 'l1', 'variance')

    Returns:
        重要性分数张量 (out_features,)
    """
    if method == 'l2':
        # L2 范数
        importance = torch.norm(layer.weight.data, p=2, dim=1)
    elif method == 'l1':
        # L1 范数
        importance = torch.norm(layer.weight.data, p=1, dim=1)
    elif method == 'variance':
        # 方差
        importance = torch.var(layer.weight.data, dim=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    return importance


if __name__ == '__main__':
    # 测试剪枝功能
    print("=" * 70)
    print("")
    print("=" * 70)

    # 创建简单的 MLP
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
            )
            self.classifier = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.features(x)
            return self.classifier(x)

    # 创建模型
    model = SimpleMLP()
    print(f"\n: {sum(p.numel() for p in model.parameters()):,}")

    # 创建剪枝器
    pruner = StructuredPruning(model)

    # 测试按比例剪枝
    print("\n features.0  (50% )")
    keep_indices = pruner.prune_mlp_by_ratio('features.0', prune_ratio=0.5)
    print(f": {len(keep_indices)} / 512")

    # 剪枝连接层
    pruner.prune_connected_layers('features.0', 'features.2', keep_indices)

    # 打印剪枝后信息
    info = pruner.get_model_info()
    print(f"\n:")
    print(f"  : {info['total_params']:,}")
    print(f"  : {info['model_size_mb']:.2f} MB")
    print(f"  : {info['sparsity']:.2%}")

    # 测试前向传播
    dummy_input = torch.randn(4, 784)
    output = model(dummy_input)
    print(f"\n:")
    print(f"  : {dummy_input.shape}")
    print(f"  : {output.shape}")

    print("\n[SUCCESS] !")


# 为了向后兼容，提供 StructuredPruner 别名
StructuredPruner = StructuredPruning
