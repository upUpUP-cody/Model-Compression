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

    print("\n:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 加载数据
    print("\n" + "-" * 70)
    print(" MNIST ...")
    train_loader, test_loader = get_mnist_loaders(
        batch_size=config['batch_size'],
        num_workers=0  # CPU 环境使用 0
    )
    print(f"[OK] : {len(train_loader)}")
    print(f"[OK] : {len(test_loader)}")

    # 创建模型
    print("\n" + "-" * 70)
    print(" Dense Baseline ...")
    model = create_mnist_baseline(
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate']
    )
    print(f"[OK] : {model.get_num_parameters():,}")
    print(f"[OK] : {model.get_model_size_mb():.2f} MB")

    # 创建训练器
    print("\n" + "-" * 70)
    print(f" (Device: {config['device']})...")
    trainer = ModelTrainer(
        model=model,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 训练模型
    print("\n" + "-" * 70)
    print("...")
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
    print("!")
    print("=" * 70)

    final_train_acc = history['train_acc'][-1]
    final_test_acc = history['test_acc'][-1]
    best_test_acc = max(history['test_acc'])

    print(f"\n:")
    print(f"  : {final_train_acc:.2f}%")
    print(f"  : {final_test_acc:.2f}%")
    print(f"  : {best_test_acc:.2f}%")
    print(f"  : {train_time:.2f}  ({train_time/60:.2f} )")
    print(f"  : {train_time/config['epochs']:.2f} ")

    # 验证成功标准
    print("\n" + "-" * 70)
    print(":")
    success = True

    if best_test_acc >= 98.0:
        print(f"  [PASS]  >= 98%: {best_test_acc:.2f}%")
    else:
        print(f"  [FAIL]  >= 98%: {best_test_acc:.2f}%")
        success = False

    if success:
        print("\n[SUCCESS]  A.1 ! Dense Baseline !")
    else:
        print("\n[WARNING] ")

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

    print(f"\n: {history_path}")

    return history, best_test_acc


if __name__ == '__main__':
    main()
