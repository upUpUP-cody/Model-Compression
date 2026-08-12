"""
测试结构化剪枝在训练好的模型上的效果
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import copy
from src.models.dense_baseline import create_mnist_baseline, ModelTrainer
from src.pruning.structured_pruning import StructuredPruning, compute_layer_importance
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
    print("结构化剪枝实验 - One-shot Pruning")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据
    print("\n加载数据...")
    _, test_loader = get_mnist_loaders(batch_size=128, num_workers=0)

    # 加载训练好的模型
    print("\n加载训练好的模型...")
    model = create_mnist_baseline()
    checkpoint = torch.load('./checkpoints/mnist_dense_baseline.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # 评估原始模型
    print("\n评估原始模型...")
    original_acc = evaluate_model(model, test_loader, device)
    original_params = sum(p.numel() for p in model.parameters())

    print(f"  准确率: {original_acc:.2f}%")
    print(f"  参数量: {original_params:,}")

    # 测试不同剪枝比例
    prune_ratios = [0.3, 0.5, 0.7, 0.9]

    print("\n" + "=" * 70)
    print("One-shot Pruning 测试")
    print("=" * 70)

    results = []

    for prune_ratio in prune_ratios:
        print(f"\n剪枝比例: {prune_ratio:.1%}")
        print("-" * 70)

        # 深拷贝模型
        pruned_model = copy.deepcopy(model)
        pruner = StructuredPruning(pruned_model)

        # 计算每层的重要性分数并剪枝
        # 只剪枝中间隐藏层，保留输入输出层
        # 模型结构: Linear(0) -> BN(1) -> ReLU(2) -> Dropout(3) -> Linear(4) -> ...
        layers_to_prune = [
            ('features.0', 'features.4'),   # 第一层 -> 第二层
            ('features.4', 'features.8'),   # 第二层 -> 第三层
            ('features.8', 'classifier'),   # 第三层 -> 分类器
        ]

        for layer_name, next_layer_name in layers_to_prune:
            try:
                # 获取层
                layer = pruner._get_layer_by_name(layer_name)

                # 计算重要性
                importance = compute_layer_importance(layer, method='l2')

                # 按比例剪枝
                keep_indices = pruner.prune_mlp_by_ratio(
                    layer_name,
                    prune_ratio,
                    importance
                )

                # 剪枝线性层和其 BatchNorm
                pruner.prune_linear_block(
                    layer_name,
                    keep_indices,
                    next_layer_name
                )

                print(f"  {layer_name}: 保留 {len(keep_indices)} 个神经元")

            except Exception as e:
                print(f"  [WARNING] 剪枝 {layer_name} 失败: {e}")

        # 获取剪枝后模型信息
        info = pruner.get_model_info()

        # 评估剪枝后模型 (无恢复)
        pruned_acc = evaluate_model(pruned_model, test_loader, device)

        # 计算压缩率
        compression_ratio = original_params / info['total_params']
        acc_drop = original_acc - pruned_acc

        print(f"\n  剪枝后参数量: {info['total_params']:,}")
        print(f"  压缩率: {compression_ratio:.2f}x")
        print(f"  剪枝后准确率: {pruned_acc:.2f}%")
        print(f"  准确率下降: {acc_drop:.2f}%")

        results.append({
            'prune_ratio': prune_ratio,
            'params': info['total_params'],
            'compression_ratio': compression_ratio,
            'accuracy': pruned_acc,
            'acc_drop': acc_drop
        })

    # 打印汇总结果
    print("\n" + "=" * 70)
    print("实验汇总")
    print("=" * 70)

    print(f"\n{'剪枝比例':<10} {'参数量':<15} {'压缩率':<10} {'准确率':<10} {'下降':<10}")
    print("-" * 70)
    print(f"{'原始':<10} {original_params:<15,} {'1.00x':<10} {original_acc:<10.2f}% {'0.00%':<10}")

    for r in results:
        print(f"{r['prune_ratio']:<10.0%} {r['params']:<15,} {r['compression_ratio']:<10.2f}x "
              f"{r['accuracy']:<10.2f}% {r['acc_drop']:<10.2f}%")

    print("\n关键发现:")
    print(f"  - 30% 剪枝后准确率: {results[0]['accuracy']:.2f}%")
    print(f"  - 50% 剪枝后准确率: {results[1]['accuracy']:.2f}%")
    print(f"  - 70% 剪枝后准确率: {results[2]['accuracy']:.2f}%")
    print(f"  - 90% 剪枝后准确率: {results[3]['accuracy']:.2f}%")
    print(f"\n  注意: 这是 one-shot pruning 无恢复的结果")
    print(f"       后续步骤将实现恢复策略来提升剪枝后性能")

    # 保存结果
    import json
    result_path = './results/pruning_oneshot_results.json'
    with open(result_path, 'w') as f:
        json.dump({
            'original': {
                'accuracy': original_acc,
                'params': original_params
            },
            'pruned_results': results
        }, f, indent=2)

    print(f"\n结果已保存至: {result_path}")


if __name__ == '__main__':
    main()
