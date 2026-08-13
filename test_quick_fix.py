"""快速测试剪枝+恢复是否工作"""
import torch
from torchvision import datasets, transforms
from src.models.dense_baseline import DenseBaseline
from src.pruning.structured_pruning import StructuredPruner
from src.recovery.reconstruction import quick_recovery
import copy

# 设备
device = 'cpu'
print(f"Using device: {device}")

# 加载数据（只用小批量测试）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# 只用小批量数据快速测试
train_subset = torch.utils.data.Subset(train_dataset, range(1000))
test_subset = torch.utils.data.Subset(test_dataset, range(200))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=200)

# 加载模型
model = DenseBaseline().to(device)
checkpoint = torch.load('checkpoints/mnist_dense_baseline.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"[OK] Model loaded, baseline: {checkpoint['test_acc']:.2f}%")

# 深拷贝模型
model_copy = copy.deepcopy(model)
print(f"Original params: {sum(p.numel() for p in model_copy.parameters()):,}")

# 剪枝
pruner = StructuredPruner(model_copy)
pruned_params = pruner.prune_uniform(0.3)
print(f"After pruning (30%): {pruned_params:,} params")

# 评估剪枝后
model_copy.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model_copy(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
pruned_acc = 100.0 * correct / total
print(f"Pruned accuracy: {pruned_acc:.2f}%")

# 快速恢复（只训练1轮测试）
print("\nStarting recovery (1 epoch)...")
_, history = quick_recovery(
    model_copy,
    train_loader,
    test_loader,
    epochs=1,
    device=device,
    verbose=True
)

print(f"\n[SUCCESS] Test passed!")
print(f"  Best accuracy: {history['best_validation_accuracy']:.2f}%")
