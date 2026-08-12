"""
MNIST Dense Baseline 实验
训练并评估 MLP 模型作为基准
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import time
from src.models.dense_baseline import create_mnist_baseline, ModelTrainer
from src.utils.data_loader import get_mnist_loaders


def main():
    print("=" * 70)
    print("MNIST Dense Baseline Experiment - Step A.1")
    print("=" * 70)

    # 配置
    config = {
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.2,
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_path': './checkpoints/mnist_dense_baseline.pth'
    }

    print("\n配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 加载数据
    print("\n" + "-" * 70)
    print("加载 MNIST 数据集...")
    train_loader, test_loader = get_mnist_loaders(
        batch_size=config['batch_size'],
        num_workers=0  # CPU 环境使用 0
    )
    print(f"[OK] 训练批次数: {len(train_loader)}")
    print(f"[OK] 测试批次数: {len(test_loader)}")

    # 创建模型
    print("\n" + "-" * 70)
    print("创建 Dense Baseline 模型...")
    model = create_mnist_baseline(
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate']
    )
    print(f"[OK] 参数量: {model.get_num_parameters():,}")
    print(f"[OK] 模型大小: {model.get_model_size_mb():.2f} MB")

    # 创建训练器
    print("\n" + "-" * 70)
    print(f"初始化训练器 (Device: {config['device']})...")
    trainer = ModelTrainer(
        model=model,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 训练模型
    print("\n" + "-" * 70)
    print("开始训练...")
    start_time = time.time()

    history = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config['epochs'],
        verbose=True,
        save_path=config['save_path']
    )

    train_time = time.time() - start_time

    # 最终评估
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)

    final_train_acc = history['train_acc'][-1]
    final_test_acc = history['test_acc'][-1]
    best_test_acc = max(history['test_acc'])

    print(f"\n最终结果:")
    print(f"  训练准确率: {final_train_acc:.2f}%")
    print(f"  测试准确率: {final_test_acc:.2f}%")
    print(f"  最佳测试准确率: {best_test_acc:.2f}%")
    print(f"  训练时间: {train_time:.2f} 秒 ({train_time/60:.2f} 分钟)")
    print(f"  平均每轮: {train_time/config['epochs']:.2f} 秒")

    # 验证成功标准
    print("\n" + "-" * 70)
    print("验证成功标准:")
    success = True

    if best_test_acc >= 98.0:
        print(f"  [PASS] 测试准确率 >= 98%: {best_test_acc:.2f}%")
    else:
        print(f"  [FAIL] 测试准确率 >= 98%: {best_test_acc:.2f}%")
        success = False

    if success:
        print("\n[SUCCESS] 步骤 A.1 完成! Dense Baseline 训练成功!")
    else:
        print("\n[WARNING] 未达到预期标准，可能需要调整超参数")

    # 保存训练历史
    import json
    history_path = './results/mnist_baseline_history.json'
    Path(history_path).parent.mkdir(parents=True, exist_ok=True)

    with open(history_path, 'w') as f:
        json.dump({
            'config': config,
            'history': history,
            'final_results': {
                'train_acc': final_train_acc,
                'test_acc': final_test_acc,
                'best_test_acc': best_test_acc,
                'train_time': train_time
            }
        }, f, indent=2)

    print(f"\n训练历史已保存至: {history_path}")

    return history, best_test_acc


if __name__ == '__main__':
    main()
