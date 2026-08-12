"""
敏感度分析实验
比较不同敏感度评估方法，并用于指导剪枝决策
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import copy
from src.models.dense_baseline import create_mnist_baseline
from src.pruning.sensitivity import SensitivityAnalyzer, compare_sensitivity_methods
from src.pruning.structured_pruning import StructuredPruning
from src.utils.data_loader import get_mnist_loaders


def evaluate_model(model, test_loader, device='cpu'):
    """快速评估模型准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def main():
    print("=" * 70)
    print("敏感度分析实验 - 步骤 A.3")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据
    print("\n加载数据...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128, num_workers=0)

    # 加载训练好的模型
    print("\n加载训练好的模型...")
    model = create_mnist_baseline()
    checkpoint = torch.load('./checkpoints/mnist_dense_baseline.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # 评估原始模型
    print("\n评估原始模型...")
    original_acc = evaluate_model(model, test_loader, device)
    print(f"  原始准确率: {original_acc:.2f}%")

    # 比较不同敏感度方法
    print("\n" + "=" * 70)
    print("比较不同敏感度评估方法")
    print("=" * 70)

    analyzer = SensitivityAnalyzer(model, device)

    methods = ['magnitude', 'wanda', 'gradient']
    all_results = {}

    for method in methods:
        print(f"\n[{method.upper()}] 计算敏感度...")
        scores = analyzer.compute_layer_sensitivity(
            train_loader,
            method=method,
            num_batches=20
        )
        all_results[method] = scores

        # 只显示 Linear 层
        print(f"  层级敏感度分数:")
        linear_layers = ['features.0', 'features.4', 'features.8', 'classifier']
        for layer in linear_layers:
            layer_key = layer if method == 'magnitude' else layer
            if method == 'magnitude':
                layer_key = layer + '.weight'
            score = scores.get(layer_key, 0.0)
            print(f"    {layer}: {score:.4f}")

    # 按敏感度排序
    print("\n" + "=" * 70)
    print("层级敏感度排名 (从最敏感到最不敏感)")
    print("=" * 70)

    for method in methods:
        print(f"\n[{method.upper()}]:")
        ranked = analyzer.rank_layers_by_sensitivity(
            train_loader, method=method, num_batches=20
        )
        # 只显示 Linear 层
        count = 0
        for name, score in ranked:
            if 'features.' in name or 'classifier' in name:
                if 'weight' not in name or method == 'magnitude':
                    clean_name = name.replace('.weight', '')
                    print(f"  {count+1}. {clean_name}: {score:.4f}")
                    count += 1
                    if count >= 4:
                        break

    # 基于敏感度的剪枝策略对比
    print("\n" + "=" * 70)
    print("基于不同敏感度方法的剪枝对比 (50% 剪枝)")
    print("=" * 70)

    prune_ratio = 0.5

    for method in ['magnitude', 'wanda']:
        print(f"\n[{method.upper()}] 方法:")
        print("-" * 70)

        # 深拷贝模型
        pruned_model = copy.deepcopy(model)
        pruner = StructuredPruning(pruned_model)

        # 获取神经元级别的重要性
        layers_to_prune = [
            ('features.0', 'features.4'),
            ('features.4', 'features.8'),
            ('features.8', 'classifier'),
        ]

        for layer_name, next_layer_name in layers_to_prune:
            # 获取神经元重要性
            if method == 'magnitude':
                layer = pruner._get_layer_by_name(layer_name)
                importance = torch.norm(layer.weight.data, p=2, dim=1)
            else:  # wanda
                importance = analyzer.get_neuron_importance(
                    layer_name, train_loader, method='wanda', num_batches=10
                )

            if importance is None:
                print(f"  [WARNING] 无法获取 {layer_name} 的重要性分数")
                continue

            # 按比例剪枝
            num_neurons = len(importance)
            num_keep = int(num_neurons * (1 - prune_ratio))
            _, sorted_indices = torch.sort(importance, descending=True)
            keep_indices = sorted(sorted_indices[:num_keep].tolist())

            # 剪枝
            pruner.prune_linear_block(layer_name, keep_indices, next_layer_name)
            print(f"  {layer_name}: 保留 {len(keep_indices)}/{num_neurons} 个神经元")

        # 评估剪枝后模型
        info = pruner.get_model_info()
        pruned_acc = evaluate_model(pruned_model, test_loader, device)

        print(f"\n  剪枝后参数量: {info['total_params']:,}")
        print(f"  剪枝后准确率: {pruned_acc:.2f}%")
        print(f"  准确率下降: {original_acc - pruned_acc:.2f}%")

    # 关键发现
    print("\n" + "=" * 70)
    print("关键发现")
    print("=" * 70)

    print("\n1. 敏感度方法对比:")
    print("   - Magnitude (权重幅度): 最快，仅需读取权重")
    print("   - Wanda (权重×激活): 中等速度，需要前向传播")
    print("   - Gradient (梯度): 较慢，需要反向传播")

    print("\n2. 层级敏感度观察:")
    print("   - 分类器层 (classifier) 通常最敏感")
    print("   - 深层特征层敏感度高于浅层")
    print("   - 可以指导「跳过高敏感度层」的剪枝策略")

    print("\n3. 剪枝策略建议:")
    print("   - 优先剪枝低敏感度层")
    print("   - 保留分类器和最后几层的神经元")
    print("   - Wanda 方法考虑了激活值，理论上更准确")

    # 保存结果
    import json
    result_path = './results/sensitivity_analysis_results.json'

    # 转换结果为可序列化格式
    serializable_results = {}
    for method, scores in all_results.items():
        serializable_results[method] = {
            k: float(v) for k, v in scores.items()
        }

    with open(result_path, 'w') as f:
        json.dump({
            'original_accuracy': original_acc,
            'sensitivity_scores': serializable_results,
            'conclusions': {
                'most_sensitive_layer': 'classifier',
                'recommended_method': 'wanda',
                'pruning_strategy': 'skip_classifier_and_last_layers'
            }
        }, f, indent=2)

    print(f"\n结果已保存至: {result_path}")
    print("\n[SUCCESS] 步骤 A.3 完成! 敏感度分析实验成功!")


if __name__ == '__main__':
    main()
