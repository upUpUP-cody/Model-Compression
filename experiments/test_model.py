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
    print("模型测试")
    print("=" * 70)

    # 加载数据
    print("\n加载测试数据...")
    _, test_loader = get_mnist_loaders(batch_size=128, num_workers=0)
    print(f"[OK] 测试批次数: {len(test_loader)}")

    # 创建模型
    print("\n创建模型...")
    model = create_mnist_baseline()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 加载权重
    print(f"\n加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[OK] 检查点轮次: {checkpoint['epoch']}")
    print(f"[OK] 检查点准确率: {checkpoint['test_acc']:.2f}%")

    # 评估模型
    print("\n" + "-" * 70)
    print("评估模型...")
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
    print(f"\n测试结果:")
    print(f"  准确率: {accuracy:.2f}%")
    print(f"  正确数: {correct}/{total}")
    print(f"  推理时间: {inference_time:.2f} 秒")
    print(f"  平均每样本: {inference_time/total*1000:.2f} ms")

    # 模型信息
    print(f"\n模型信息:")
    print(f"  参数量: {model.get_num_parameters():,}")
    print(f"  模型大小: {model.get_model_size_mb():.2f} MB")

    return accuracy


if __name__ == '__main__':
    checkpoint_path = './checkpoints/mnist_dense_baseline.pth'
    test_model(checkpoint_path)
