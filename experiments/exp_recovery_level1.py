"""
完整的剪枝+恢复实验
测试不同剪枝比例下 Level 1 重建恢复的效果
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import copy
from src.models.dense_baseline import create_mnist_baseline
from src.pruning.structured_pruning import StructuredPruning
from src.recovery.reconstruction import ReconstructionRecovery
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


def prune_and_recover(
    original_model,
    prune_ratio,
    train_loader,
    test_loader,
    recovery_epochs=10,
    device='cpu'
):
    """
    剪枝并恢复模型

    Returns:
        dict: 包含各阶段准确率的字典
    """
    # 深拷贝模型
    pruned_model = copy.deepcopy(original_model)
    pruner = StructuredPruning(pruned_model)

    # 剪枝
    layers_to_prune = [
        ('features.0', 'features.4'),
        ('features.4', 'features.8'),
        ('features.8', 'classifier'),
    ]

    for layer_name, next_layer_name in layers_to_prune:
        layer = pruner._get_layer_by_name(layer_name)
        importance = torch.norm(layer.weight.data, p=2, dim=1)
        keep_indices = pruner.prune_mlp_by_ratio(layer_name, prune_ratio, importance)
        pruner.prune_linear_block(layer_name, keep_indices, next_layer_name)

    # 获取模型信息
    info = pruner.get_model_info()

    # 评估剪枝后模型（无恢复）
    pruned_acc = evaluate_model(pruned_model, test_loader, device)

    # Level 1 恢复
    recoverer = ReconstructionRecovery(
        pruned_model,
        device=device,
        learning_rate=0.001
    )

    history = recoverer.recover(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=recovery_epochs,
        verbose=False
    )

    return {
        'params': info['total_params'],
        'pruned_acc': pruned_acc,
        'recovered_acc': history['best_test_acc'],
        'history': history
    }


def main():
    print("=" * 70)
    print("剪枝 + 重建恢复完整实验 (步骤 C.2)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据
    print("\n加载数据...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128, num_workers=0)

    # 加载训练好的模型
    print("加载训练好的模型...")
    model = create_mnist_baseline()
    checkpoint = torch.load('./checkpoints/mnist_dense_baseline.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # 评估原始模型
    print("评估原始模型...")
    original_acc = evaluate_model(model, test_loader, device)
    original_params = sum(p.numel() for p in model.parameters())

    print(f"  原始准确率: {original_acc:.2f}%")
    print(f"  原始参数量: {original_params:,}")

    # 测试不同剪枝比例
    prune_ratios = [0.3, 0.5, 0.7, 0.9]
    recovery_epochs = 10

    print("\n" + "=" * 70)
    print(f"测试不同剪枝比例 (恢复训练 {recovery_epochs} 轮)")
    print("=" * 70)

    results = []

    for prune_ratio in prune_ratios:
        print(f"\n剪枝比例: {prune_ratio:.0%}")
        print("-" * 70)

        result = prune_and_recover(
            original_model=model,
            prune_ratio=prune_ratio,
            train_loader=train_loader,
            test_loader=test_loader,
            recovery_epochs=recovery_epochs,
            device=device
        )

        compression_ratio = original_params / result['params']
        recovery_gain = result['recovered_acc'] - result['pruned_acc']

        print(f"  剪枝后参数量: {result['params']:,}")
        print(f"  压缩率: {compression_ratio:.2f}x")
        print(f"  剪枝后准确率 (无恢复): {result['pruned_acc']:.2f}%")
        print(f"  恢复后准确率: {result['recovered_acc']:.2f}%")
        print(f"  恢复提升: {recovery_gain:.2f}%")

        results.append({
            'prune_ratio': prune_ratio,
            'params': result['params'],
            'compression_ratio': compression_ratio,
            'pruned_acc': result['pruned_acc'],
            'recovered_acc': result['recovered_acc'],
            'recovery_gain': recovery_gain
        })

    # 对比表格
    print("\n" + "=" * 70)
    print("实验汇总")
    print("=" * 70)

    print(f"\n{'剪枝比例':<10} {'参数量':<15} {'压缩率':<10} {'剪枝后':<10} {'恢复后':<10} {'恢复提升':<10}")
    print("-" * 70)
    print(f"{'原始':<10} {original_params:<15,} {'1.00x':<10} {original_acc:<10.2f}% {'-':<10} {'-':<10}")

    for r in results:
        print(f"{r['prune_ratio']:<10.0%} {r['params']:<15,} {r['compression_ratio']:<10.2f}x "
              f"{r['pruned_acc']:<10.2f}% {r['recovered_acc']:<10.2f}% {r['recovery_gain']:<10.2f}%")

    # 关键发现
    print("\n" + "=" * 70)
    print("关键发现")
    print("=" * 70)

    print("\n1. 恢复效果对比:")
    for r in results:
        status = "优秀" if r['recovered_acc'] >= 98.0 else "良好" if r['recovered_acc'] >= 95.0 else "一般"
        print(f"   {r['prune_ratio']:.0%} 剪枝: {r['pruned_acc']:.2f}% → {r['recovered_acc']:.2f}% "
              f"(+{r['recovery_gain']:.2f}%) [{status}]")

    print("\n2. Level 1 重建恢复的价值:")
    print(f"   - 30% 剪枝: 恢复提升 {results[0]['recovery_gain']:.2f}%")
    print(f"   - 50% 剪枝: 恢复提升 {results[1]['recovery_gain']:.2f}%")
    print(f"   - 70% 剪枝: 恢复提升 {results[2]['recovery_gain']:.2f}%")
    print(f"   - 90% 剪枝: 恢复提升 {results[3]['recovery_gain']:.2f}%")

    print("\n3. 压缩与性能权衡:")
    best_tradeoff = max(results, key=lambda x: x['recovered_acc'] - (100 - x['recovered_acc']))
    print(f"   最佳权衡点: {best_tradeoff['prune_ratio']:.0%} 剪枝")
    print(f"   - 准确率: {best_tradeoff['recovered_acc']:.2f}%")
    print(f"   - 压缩率: {best_tradeoff['compression_ratio']:.2f}x")
    print(f"   - 性能损失: {original_acc - best_tradeoff['recovered_acc']:.2f}%")

    print("\n4. 与 One-shot Pruning 对比:")
    print(f"   50% 剪枝:")
    print(f"   - One-shot: 67.87% (损失 30.56%)")
    print(f"   - Level 1 恢复: {results[1]['recovered_acc']:.2f}% (损失 {original_acc - results[1]['recovered_acc']:.2f}%)")
    print(f"   - 恢复策略使性能提升 {results[1]['recovery_gain']:.2f}%!")

    # 保存结果
    import json
    result_path = './results/recovery_level1_results.json'

    with open(result_path, 'w') as f:
        json.dump({
            'original': {
                'accuracy': original_acc,
                'params': original_params
            },
            'recovery_epochs': recovery_epochs,
            'results': results,
            'conclusions': {
                'best_tradeoff': {
                    'prune_ratio': best_tradeoff['prune_ratio'],
                    'accuracy': best_tradeoff['recovered_acc'],
                    'compression': best_tradeoff['compression_ratio']
                }
            }
        }, f, indent=2)

    print(f"\n结果已保存至: {result_path}")
    print("\n[SUCCESS] 步骤 C.2 完成! Level 1 重建恢复实验成功!")


if __name__ == '__main__':
    main()
