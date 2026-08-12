"""
Capability Frontier Profiling

根据论文 Section 3.1 - Self-Diagnosis 实现前沿分析
自动探索不同剪枝比例下的性能边界
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

from ..pruning.structured_pruning import StructuredPruner
from ..recovery.reconstruction import quick_recovery


class FrontierProfiler:
    """前沿分析器 - 探索剪枝比例和性能的帕累托前沿"""

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.pruner = StructuredPruner(model)

    def profile(
        self,
        train_loader,
        test_loader,
        prune_ratios: Optional[List[float]] = None,
        recovery_epochs: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        探索前沿

        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            prune_ratios: 要测试的剪枝比例列表
            recovery_epochs: 每个点的恢复训练轮数
            verbose: 是否打印详细信息

        Returns:
            前沿分析结果字典
        """
        if prune_ratios is None:
            # 默认探索点: 10%, 20%, ..., 90%
            prune_ratios = [i * 0.1 for i in range(1, 10)]

        # 获取原始性能
        original_acc = self._evaluate(test_loader)
        original_params = sum(p.numel() for p in self.model.parameters())

        if verbose:
            print(f"\n{'='*60}")
            print(f"Capability Frontier Profiling")
            print(f"{'='*60}")
            print(f"Original Model: {original_acc:.2f}% accuracy, {original_params:,} params")
            print(f"Exploring {len(prune_ratios)} pruning ratios...")
            print(f"{'='*60}\n")

        frontier_points = []

        for ratio in prune_ratios:
            if verbose:
                print(f"\n[{ratio:.0%} Pruning]")

            # 深拷贝模型（避免修改原始模型）
            import copy
            model_copy = copy.deepcopy(self.model)
            pruner_copy = StructuredPruner(model_copy)

            # 剪枝
            pruned_params = pruner_copy.prune_uniform(ratio)

            # 评估剪枝后准确率（使用拷贝的模型）
            model_copy.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model_copy(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            pruned_acc = 100.0 * correct / total

            if verbose:
                print(f"  After pruning: {pruned_acc:.2f}%")

            # 快速恢复（使用拷贝的模型）
            if recovery_epochs > 0:
                _, history = quick_recovery(
                    model_copy,
                    train_loader,
                    test_loader,
                    epochs=recovery_epochs,
                    device=self.device,
                    verbose=False
                )
                recovered_acc = history['best_test_acc']
                if verbose:
                    print(f"  After recovery: {recovered_acc:.2f}% (+{recovered_acc - pruned_acc:.2f}%)")
            else:
                recovered_acc = pruned_acc

            # 记录前沿点
            compression_ratio = original_params / pruned_params
            frontier_points.append({
                'prune_ratio': ratio,
                'params': pruned_params,
                'compression_ratio': compression_ratio,
                'pruned_acc': pruned_acc,
                'recovered_acc': recovered_acc,
                'accuracy_drop': original_acc - recovered_acc,
                'efficiency_score': recovered_acc / compression_ratio  # 性能/压缩比
            })

            # 不需要恢复，因为我们使用了深拷贝

        # 分析前沿
        analysis = self._analyze_frontier(frontier_points, original_acc, verbose)

        result = {
            'original': {
                'accuracy': original_acc,
                'params': original_params
            },
            'frontier_points': frontier_points,
            'analysis': analysis
        }

        return result

    def _analyze_frontier(
        self,
        points: List[Dict],
        original_acc: float,
        verbose: bool = True
    ) -> Dict:
        """分析前沿特性"""

        # 找到最优点
        best_compression = max(points, key=lambda x: x['compression_ratio'])
        best_accuracy = max(points, key=lambda x: x['recovered_acc'])
        best_tradeoff = max(points, key=lambda x: x['efficiency_score'])

        # 找到可接受的最大压缩（损失 < 1%）
        acceptable_points = [
            p for p in points
            if p['accuracy_drop'] < 1.0
        ]
        max_acceptable = max(acceptable_points, key=lambda x: x['compression_ratio']) \
            if acceptable_points else points[0]

        # 计算前沿梯度（性能下降速率）
        gradients = []
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            acc_change = p2['recovered_acc'] - p1['recovered_acc']
            ratio_change = p2['prune_ratio'] - p1['prune_ratio']
            gradients.append(acc_change / ratio_change)

        # 找到"拐点" - 梯度变化最大的地方
        if len(gradients) > 1:
            gradient_changes = [abs(gradients[i+1] - gradients[i]) for i in range(len(gradients)-1)]
            knee_idx = gradient_changes.index(max(gradient_changes)) + 1
            knee_point = points[knee_idx]
        else:
            knee_point = points[0]

        analysis = {
            'best_compression': {
                'prune_ratio': best_compression['prune_ratio'],
                'accuracy': best_compression['recovered_acc'],
                'compression': best_compression['compression_ratio']
            },
            'best_accuracy': {
                'prune_ratio': best_accuracy['prune_ratio'],
                'accuracy': best_accuracy['recovered_acc'],
                'compression': best_accuracy['compression_ratio']
            },
            'best_tradeoff': {
                'prune_ratio': best_tradeoff['prune_ratio'],
                'accuracy': best_tradeoff['recovered_acc'],
                'compression': best_tradeoff['compression_ratio'],
                'efficiency_score': best_tradeoff['efficiency_score']
            },
            'max_acceptable': {
                'prune_ratio': max_acceptable['prune_ratio'],
                'accuracy': max_acceptable['recovered_acc'],
                'compression': max_acceptable['compression_ratio'],
                'accuracy_drop': max_acceptable['accuracy_drop']
            },
            'knee_point': {
                'prune_ratio': knee_point['prune_ratio'],
                'accuracy': knee_point['recovered_acc'],
                'compression': knee_point['compression_ratio'],
                'description': '性能下降加速的拐点'
            }
        }

        if verbose:
            print(f"\n{'='*60}")
            print("Frontier Analysis")
            print(f"{'='*60}")
            print(f"\n🎯 推荐配置:")
            print(f"  最佳权衡点: {best_tradeoff['prune_ratio']:.0%} 剪枝")
            print(f"    → {best_tradeoff['recovered_acc']:.2f}% 准确率")
            print(f"    → {best_tradeoff['compression_ratio']:.2f}x 压缩")
            print(f"\n  最大可接受压缩 (< 1% 损失):")
            print(f"    → {max_acceptable['prune_ratio']:.0%} 剪枝")
            print(f"    → {max_acceptable['recovered_acc']:.2f}% 准确率")
            print(f"    → {max_acceptable['compression_ratio']:.2f}x 压缩")
            print(f"\n  性能拐点:")
            print(f"    → {knee_point['prune_ratio']:.0%} 剪枝")
            print(f"    → {knee_point['recovered_acc']:.2f}% 准确率")
            print(f"{'='*60}\n")

        return analysis

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

    def save_results(self, results: Dict, save_path: str):
        """保存前沿分析结果"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to {save_path}")

    def plot_frontier(self, results: Dict, save_path: Optional[str] = None):
        """
        绘制前沿曲线

        需要 matplotlib，如果不可用则跳过
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠ matplotlib not available, skipping plot")
            return

        points = results['frontier_points']
        original_acc = results['original']['accuracy']

        prune_ratios = [p['prune_ratio'] * 100 for p in points]
        compressions = [p['compression_ratio'] for p in points]
        pruned_accs = [p['pruned_acc'] for p in points]
        recovered_accs = [p['recovered_acc'] for p in points]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 左图: 准确率 vs 剪枝比例
        ax1.plot(prune_ratios, pruned_accs, 'o--', label='After Pruning', color='#F97316')
        ax1.plot(prune_ratios, recovered_accs, 'o-', label='After Recovery', color='#38BDF8')
        ax1.axhline(y=original_acc, color='#111827', linestyle=':', label='Original')
        ax1.set_xlabel('Pruning Ratio (%)')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy vs Pruning Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右图: 准确率 vs 压缩率 (帕累托前沿)
        ax2.plot(compressions, recovered_accs, 'o-', color='#4FD1C5', linewidth=2)
        ax2.axhline(y=original_acc, color='#111827', linestyle=':', label='Original')
        ax2.set_xlabel('Compression Ratio (x)')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Pareto Frontier: Accuracy vs Compression')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


def find_optimal_pruning_ratio(
    model: nn.Module,
    train_loader,
    test_loader,
    target_accuracy_drop: float = 1.0,
    recovery_epochs: int = 5
) -> Tuple[float, Dict]:
    """
    自动找到最优剪枝比例

    Args:
        model: 要分析的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        target_accuracy_drop: 可接受的准确率下降（默认1%）
        recovery_epochs: 恢复训练轮数

    Returns:
        (最优剪枝比例, 完整分析结果)
    """
    profiler = FrontierProfiler(model)
    results = profiler.profile(
        train_loader,
        test_loader,
        recovery_epochs=recovery_epochs,
        verbose=True
    )

    # 找到满足目标的最大剪枝比例
    optimal_ratio = results['analysis']['max_acceptable']['prune_ratio']

    return optimal_ratio, results
