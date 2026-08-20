"""
实验: Lottery Ticket Proposal

基于敏感度生成和评估候选剪枝方案
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torchvision import datasets, transforms

from src.models.dense_baseline import DenseBaseline
from src.frontier.proposal import LotteryTicketProposer


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

    # 创建提议器
    proposer = LotteryTicketProposer(model, device)

    # 测试不同的目标压缩率
    target_compressions = [2.0, 4.0, 8.0]

    all_results = {}

    for target_comp in target_compressions:
        print(f"\n{'#'*70}")
        print(f"# Target Compression: {target_comp:.1f}x")
        print(f"{'#'*70}")

        # 生成提议
        proposals = proposer.propose(
            train_loader,
            target_compression=target_comp,
            method='wanda',
            sensitivity_samples=100,
            num_proposals=3,
            verbose=True
        )

        # 保存提议
        proposer.save_proposals(
            proposals,
            f'results/proposals_compression_{target_comp:.0f}x.json'
        )

        # 评估提议
        print(f"\n{'='*60}")
        print(f"Evaluating proposals (compression {target_comp:.1f}x)...")
        print(f"{'='*60}")

        results = proposer.evaluate_proposals(
            proposals,
            train_loader,
            test_loader,
            recovery_epochs=10,
            verbose=True
        )

        all_results[f'{target_comp:.1f}x'] = results

    # 打印总结
    print(f"\n{'='*70}")
    print("Overall Summary")
    print(f"{'='*70}\n")

    for comp, results in all_results.items():
        print(f"\n Target Compression: {comp}")
        print("-" * 70)

        if results:
            print(f"{'Strategy':>15} {'Actual Comp':>12} {'Pruned Acc':>12} {'Recovered Acc':>14} {'Gain':>8}")
            print("-" * 70)

            for r in results:
                print(f"{r['strategy']:>15} "
                      f"{r['actual_compression']:>11.2f}x "
                      f"{r['pruned_accuracy']:>11.2f}% "
                      f"{r['recovered_accuracy']:>13.2f}% "
                      f"{r['recovery_gain']:>7.2f}%")

            # 最佳方案
            best = results[0]
            print(f"\n[BEST] Best strategy: {best['strategy']}")
            print(f"   -> {best['recovered_accuracy']:.2f}% accuracy")
            print(f"   -> {best['actual_compression']:.2f}x compression")
        else:
            print("  [WARN] No successful proposals")

    # 保存完整结果
    import json
    with open('results/lottery_ticket_proposals_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("[OK] Lottery Ticket Proposal Complete!")
    print(f"{'='*70}\n")

    print(" Generated files:")
    print("  - results/proposals_compression_2x.json")
    print("  - results/proposals_compression_4x.json")
    print("  - results/proposals_compression_8x.json")
    print("  - results/lottery_ticket_proposals_summary.json")


if __name__ == '__main__':
    main()
