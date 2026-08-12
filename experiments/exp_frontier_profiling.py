"""
实验: Capability Frontier Profiling

探索 MNIST Dense Baseline 的剪枝性能前沿
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torchvision import datasets, transforms

from src.models.dense_baseline import DenseBaseline
from src.frontier.profiling import FrontierProfiler


def main():
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

    # 加载训练好的模型
    model = DenseBaseline().to(device)
    checkpoint = torch.load('checkpoints/mnist_dense_baseline.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("[OK] Model loaded")
    print(f"  Baseline accuracy: {checkpoint['test_acc']:.2f}%")

    # 创建前沿分析器
    profiler = FrontierProfiler(model, device)

    # 探索前沿
    # 测试点: 10%, 20%, 30%, ..., 90%
    prune_ratios = [i * 0.1 for i in range(1, 10)]

    print(f"\n{'='*60}")
    print("Starting Frontier Profiling")
    print(f"Testing {len(prune_ratios)} pruning ratios: {[f'{r:.0%}' for r in prune_ratios]}")
    print(f"Recovery epochs per point: 10")
    print(f"{'='*60}\n")

    results = profiler.profile(
        train_loader,
        test_loader,
        prune_ratios=prune_ratios,
        recovery_epochs=10,  # 每个点恢复10轮
        verbose=True
    )

    # 保存结果
    profiler.save_results(results, 'results/frontier_profiling_results.json')

    # 绘制前沿曲线
    profiler.plot_frontier(results, 'results/frontier_curve.png')

    # 打印详细分析
    print("\n" + "="*60)
    print("Detailed Frontier Analysis")
    print("="*60)

    print("\n All Frontier Points:")
    print(f"{'Prune%':>8} {'Params':>10} {'Compress':>10} {'Pruned%':>10} {'Recover%':>10} {'Drop%':>8}")
    print("-" * 70)

    for point in results['frontier_points']:
        print(f"{point['prune_ratio']*100:>7.0f}% "
              f"{point['params']:>10,} "
              f"{point['compression_ratio']:>9.2f}x "
              f"{point['pruned_acc']:>9.2f}% "
              f"{point['recovered_acc']:>9.2f}% "
              f"{point['accuracy_drop']:>7.2f}%")

    # 关键发现
    analysis = results['analysis']
    print("\n" + "="*60)
    print("Key Findings")
    print("="*60)

    print("\n[Best Tradeoff]")
    bt = analysis['best_tradeoff']
    print(f"   Prune ratio: {bt['prune_ratio']:.0%}")
    print(f"   Accuracy: {bt['accuracy']:.2f}%")
    print(f"   Compression: {bt['compression']:.2f}x")
    print(f"   Efficiency score: {bt['efficiency_score']:.2f}")

    print("\n[Max Acceptable Compression (< 1% loss)]")
    ma = analysis['max_acceptable']
    print(f"   Prune ratio: {ma['prune_ratio']:.0%}")
    print(f"   Accuracy: {ma['accuracy']:.2f}%")
    print(f"   Compression: {ma['compression']:.2f}x")
    print(f"   Accuracy drop: {ma['accuracy_drop']:.2f}%")

    print("\n[Knee Point]")
    kp = analysis['knee_point']
    print(f"   Prune ratio: {kp['prune_ratio']:.0%}")
    print(f"   Accuracy: {kp['accuracy']:.2f}%")
    print(f"   Compression: {kp['compression']:.2f}x")
    print(f"   Note: {kp['description']}")

    print("\n[Best Accuracy]")
    ba = analysis['best_accuracy']
    print(f"   Prune ratio: {ba['prune_ratio']:.0%}")
    print(f"   Accuracy: {ba['accuracy']:.2f}%")
    print(f"   Compression: {ba['compression']:.2f}x")

    print("\n[Best Compression]")
    bc = analysis['best_compression']
    print(f"   Prune ratio: {bc['prune_ratio']:.0%}")
    print(f"   Accuracy: {bc['accuracy']:.2f}%")
    print(f"   Compression: {bc['compression']:.2f}x")

    print("\n" + "="*60)
    print("[OK] Frontier Profiling Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
