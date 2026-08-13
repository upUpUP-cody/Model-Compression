"""
Lottery Ticket Proposal

基于论文 Section 3.2 - Proposal Generation
智能生成候选剪枝方案（子网络提议）
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

from ..pruning.sensitivity import SensitivityAnalyzer
from ..pruning.structured_pruning import StructuredPruner


class LotteryTicketProposer:
    """彩票提议器 - 基于敏感度生成候选子网络"""

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.analyzer = SensitivityAnalyzer(model, device)
        self.pruner = StructuredPruner(model)

    def propose(
        self,
        train_loader,
        target_compression: float = 2.0,
        method: str = 'wanda',
        sensitivity_samples: int = 100,
        num_proposals: int = 3,
        verbose: bool = True
    ) -> List[Dict]:
        """
        生成候选剪枝方案

        Args:
            train_loader: 训练数据（用于敏感度分析）
            target_compression: 目标压缩率
            method: 敏感度计算方法 ('magnitude', 'wanda', 'gradient')
            sensitivity_samples: 敏感度分析样本数
            num_proposals: 生成的提议数量
            verbose: 是否打印详细信息

        Returns:
            候选方案列表
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Lottery Ticket Proposal Generation")
            print(f"{'='*60}")
            print(f"Target compression: {target_compression:.2f}x")
            print(f"Sensitivity method: {method}")
            print(f"Generating {num_proposals} proposals...")
            print(f"{'='*60}\n")

        # 1. 计算层敏感度
        if verbose:
            print("Step 1: Computing layer sensitivities...")

        sensitivities = self.analyzer.compute_layer_sensitivity(
            train_loader,
            method=method,
            num_batches=sensitivity_samples
        )

        if verbose:
            print("\nLayer Sensitivities:")
            for layer_name, score in sensitivities.items():
                print(f"  {layer_name}: {score:.4f}")

        # 2. 生成多个候选方案
        proposals = []

        for i in range(num_proposals):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Proposal {i+1}/{num_proposals}")
                print(f"{'='*60}")

            # 根据不同策略生成方案
            if i == 0:
                # 保守策略: 高敏感度层剪枝少，低敏感度层剪枝多
                strategy = "conservative"
                layer_ratios = self._conservative_strategy(sensitivities, target_compression)
            elif i == 1:
                # 激进策略: 除了最敏感的层，其他层均匀剪枝
                strategy = "aggressive"
                layer_ratios = self._aggressive_strategy(sensitivities, target_compression)
            else:
                # 平衡策略: 基于敏感度线性分配
                strategy = "balanced"
                layer_ratios = self._balanced_strategy(sensitivities, target_compression)

            if verbose:
                print(f"Strategy: {strategy}")
                print("\nProposed pruning ratios per layer:")
                for layer_name, ratio in layer_ratios.items():
                    sens = sensitivities[layer_name]
                    print(f"  {layer_name}: {ratio:.1%} (sensitivity: {sens:.4f})")

            # 计算预期参数量
            total_params = sum(p.numel() for p in self.model.parameters())
            expected_params = self._estimate_params(layer_ratios)
            actual_compression = total_params / expected_params

            if verbose:
                print(f"\nExpected compression: {actual_compression:.2f}x")

            proposals.append({
                'strategy': strategy,
                'layer_ratios': layer_ratios,
                'sensitivities': sensitivities,
                'expected_compression': actual_compression,
                'target_compression': target_compression
            })

        if verbose:
            print(f"\n{'='*60}")
            print(f"[OK] Generated {len(proposals)} proposals")
            print(f"{'='*60}\n")

        return proposals

    def _conservative_strategy(
        self,
        sensitivities: Dict[str, float],
        target_compression: float
    ) -> Dict[str, float]:
        """
        保守策略: 按敏感度反比例分配剪枝比例

        高敏感度 -> 低剪枝比例
        低敏感度 -> 高剪枝比例
        """
        # 归一化敏感度到 [0, 1]
        sens_values = np.array(list(sensitivities.values()))
        sens_min, sens_max = sens_values.min(), sens_values.max()

        if sens_max > sens_min:
            norm_sens = {
                name: (score - sens_min) / (sens_max - sens_min)
                for name, score in sensitivities.items()
            }
        else:
            norm_sens = {name: 0.5 for name in sensitivities.keys()}

        # 敏感度高 -> 剪枝比例低
        # 使用反比例关系: ratio = max_ratio * (1 - sensitivity)
        max_ratio = 0.8  # 最大剪枝80%
        min_ratio = 0.1  # 最小剪枝10%

        layer_ratios = {}
        for name, sens in norm_sens.items():
            # 敏感度越高，剪枝比例越低
            ratio = max_ratio - (max_ratio - min_ratio) * sens
            layer_ratios[name] = ratio

        # 调整以达到目标压缩率
        layer_ratios = self._adjust_to_target_compression(
            layer_ratios, target_compression
        )

        return layer_ratios

    def _aggressive_strategy(
        self,
        sensitivities: Dict[str, float],
        target_compression: float
    ) -> Dict[str, float]:
        """
        激进策略: 保护最敏感的层，其他层激进剪枝
        """
        # 找到最敏感的层
        most_sensitive = max(sensitivities, key=sensitivities.get)

        layer_ratios = {}
        for name in sensitivities.keys():
            if name == most_sensitive:
                # 最敏感的层: 保护（只剪10%）
                layer_ratios[name] = 0.1
            else:
                # 其他层: 激进剪枝（先设为60%）
                layer_ratios[name] = 0.6

        # 调整以达到目标压缩率
        layer_ratios = self._adjust_to_target_compression(
            layer_ratios, target_compression
        )

        return layer_ratios

    def _balanced_strategy(
        self,
        sensitivities: Dict[str, float],
        target_compression: float
    ) -> Dict[str, float]:
        """
        平衡策略: 基于敏感度线性分配，但不极端
        """
        # 归一化敏感度
        sens_values = np.array(list(sensitivities.values()))
        sens_min, sens_max = sens_values.min(), sens_values.max()

        if sens_max > sens_min:
            norm_sens = {
                name: (score - sens_min) / (sens_max - sens_min)
                for name, score in sensitivities.items()
            }
        else:
            norm_sens = {name: 0.5 for name in sensitivities.keys()}

        # 中等范围的剪枝比例
        max_ratio = 0.6
        min_ratio = 0.2

        layer_ratios = {}
        for name, sens in norm_sens.items():
            ratio = max_ratio - (max_ratio - min_ratio) * sens
            layer_ratios[name] = ratio

        # 调整以达到目标压缩率
        layer_ratios = self._adjust_to_target_compression(
            layer_ratios, target_compression
        )

        return layer_ratios

    def _adjust_to_target_compression(
        self,
        layer_ratios: Dict[str, float],
        target_compression: float
    ) -> Dict[str, float]:
        """调整剪枝比例以达到目标压缩率"""
        # 迭代调整
        for _ in range(10):  # 最多调整10次
            current_compression = self._estimate_compression(layer_ratios)

            if abs(current_compression - target_compression) / target_compression < 0.05:
                # 误差 < 5%, 可接受
                break

            # 计算缩放因子
            if current_compression < target_compression:
                # 需要更多压缩 -> 增加剪枝比例
                scale = 1.1
            else:
                # 压缩过度 -> 减少剪枝比例
                scale = 0.9

            # 应用缩放
            layer_ratios = {
                name: min(0.9, max(0.05, ratio * scale))
                for name, ratio in layer_ratios.items()
            }

        return layer_ratios

    def _estimate_params(self, layer_ratios: Dict[str, float]) -> int:
        """估计剪枝后的参数量"""
        total_params = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if name in layer_ratios:
                    ratio = layer_ratios[name]
                    # 保留 (1 - ratio) 的神经元
                    out_features = module.out_features
                    in_features = module.in_features

                    kept_neurons = int(out_features * (1 - ratio))
                    params = kept_neurons * in_features  # 权重
                    if module.bias is not None:
                        params += kept_neurons  # 偏置

                    total_params += params
                else:
                    # 未剪枝的层
                    total_params += sum(p.numel() for p in module.parameters())

        return max(total_params, 1)  # 避免除零

    def _estimate_compression(self, layer_ratios: Dict[str, float]) -> float:
        """估计压缩率"""
        original_params = sum(p.numel() for p in self.model.parameters())
        pruned_params = self._estimate_params(layer_ratios)
        return original_params / pruned_params

    def evaluate_proposals(
        self,
        proposals: List[Dict],
        train_loader,
        test_loader,
        recovery_epochs: int = 5,
        verbose: bool = True
    ) -> List[Dict]:
        """
        评估候选方案

        对每个提议进行实际剪枝和恢复，测量性能
        """
        if verbose:
            print(f"\n{'='*60}")
            print("Evaluating Proposals")
            print(f"{'='*60}\n")

        from ..recovery.reconstruction import quick_recovery

        results = []

        for i, proposal in enumerate(proposals):
            if verbose:
                print(f"\n[Proposal {i+1}/{len(proposals)}] {proposal['strategy']}")

            # 深拷贝整个模型（包括结构和参数）
            import copy
            model_backup = copy.deepcopy(self.model)
            original_param_count = sum(p.numel() for p in self.model.parameters())

            # 应用提议的剪枝
            try:
                pruned_params = self.pruner.prune_by_layer(proposal['layer_ratios'])
            except Exception as e:
                if verbose:
                    print(f"  [WARNING] Pruning failed: {e}")
                # 恢复模型
                self.model = model_backup
                self.pruner = StructuredPruner(self.model)
                continue

            # 评估剪枝后性能
            pruned_acc = self._evaluate(test_loader)

            if verbose:
                print(f"  After pruning: {pruned_acc:.2f}%")

            # 恢复训练
            _, history = quick_recovery(
                self.model,
                train_loader,
                test_loader,
                epochs=recovery_epochs,
                device=self.device,
                verbose=False
            )

            # 获取最佳测试准确率
            recovered_acc = history['best_validation_accuracy']

            if verbose:
                print(f"  After recovery: {recovered_acc:.2f}% (+{recovered_acc - pruned_acc:.2f}%)")

            # 计算实际压缩率
            pruned_param_count = sum(p.numel() for p in self.model.parameters())
            actual_compression = original_param_count / pruned_param_count

            results.append({
                'proposal_id': i,
                'strategy': proposal['strategy'],
                'layer_ratios': proposal['layer_ratios'],
                'expected_compression': proposal['expected_compression'],
                'actual_compression': actual_compression,
                'pruned_accuracy': pruned_acc,
                'recovered_accuracy': recovered_acc,
                'recovery_gain': recovered_acc - pruned_acc
            })

            # 恢复原始模型
            self.model = model_backup
            self.pruner = StructuredPruner(self.model)

        # 排序: 按恢复后准确率排序
        results.sort(key=lambda x: x['recovered_accuracy'], reverse=True)

        if verbose:
            print(f"\n{'='*60}")
            print("Evaluation Results (sorted by accuracy)")
            print(f"{'='*60}\n")

            for r in results:
                print(f"{r['strategy']:>12}: {r['recovered_accuracy']:.2f}% "
                      f"({r['actual_compression']:.2f}x compression)")

        return results

    def _evaluate(self, test_loader) -> float:
        """评估模型准确率"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        return 100.0 * correct / total

    def save_proposals(self, proposals: List[Dict], save_path: str):
        """保存提议"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换为可序列化格式
        serializable = []
        for p in proposals:
            serializable.append({
                k: (v if not isinstance(v, dict) or k != 'sensitivities' else
                    {name: float(score) for name, score in v.items()})
                for k, v in p.items()
            })

        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"[OK] Proposals saved to {save_path}")


def auto_propose_lottery_ticket(
    model: nn.Module,
    train_loader,
    test_loader,
    target_compression: float = 2.0,
    method: str = 'wanda',
    recovery_epochs: int = 5,
    device: str = 'cpu'
) -> Dict:
    """
    自动生成并评估彩票提议

    Args:
        model: 要分析的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        target_compression: 目标压缩率
        method: 敏感度方法
        recovery_epochs: 恢复训练轮数
        device: 设备

    Returns:
        最佳提议和完整结果
    """
    proposer = LotteryTicketProposer(model, device)

    # 生成提议
    proposals = proposer.propose(
        train_loader,
        target_compression=target_compression,
        method=method,
        num_proposals=3,
        verbose=True
    )

    # 评估提议
    results = proposer.evaluate_proposals(
        proposals,
        train_loader,
        test_loader,
        recovery_epochs=recovery_epochs,
        verbose=True
    )

    # 返回最佳提议
    best = results[0] if results else None

    return {
        'best_proposal': best,
        'all_results': results,
        'all_proposals': proposals
    }
