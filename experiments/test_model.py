"""
模型测试脚本
用于快速测试训练好的模型
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import time
from src.models.dense_baseline import create_mnist_baseline
from src.utils.data_loader import get_mnist_loaders


def test_model(checkpoint_path: str):
    """
    测试已训练的模型

    Args:
        checkpoint_path: 模型检查点路径
    """
    print("=" * 70)
    print("")
    print("=" * 70)

    # 加载数据
    print("\n...")
    _, test_loader = get_mnist_loaders(batch_size=128, num_workers=0)
    print(f"[OK] : {len(test_loader)}")

    # 创建模型
    print("\n...")
    model = create_mnist_baseline()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 加载权重
    print(f"\n: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[OK] : {checkpoint['epoch']}")
    print(f"[OK] : {checkpoint['test_acc']:.2f}%")

    # 评估模型
    print("\n" + "-" * 70)
    print("...")
    model.eval()
    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    inference_time = time.time() - start_time
    accuracy = 100.0 * correct / total

    # 打印结果
    print(f"\n:")
    print(f"  : {accuracy:.2f}%")
    print(f"  : {correct}/{total}")
    print(f"  : {inference_time:.2f} ")
    print(f"  : {inference_time/total*1000:.2f} ms")

    # 模型信息
    print(f"\n:")
    print(f"  : {model.get_num_parameters():,}")
    print(f"  : {model.get_model_size_mb():.2f} MB")

    return accuracy


if __name__ == '__main__':
    checkpoint_path = './checkpoints/mnist_dense_baseline.pth'
    test_model(checkpoint_path)
