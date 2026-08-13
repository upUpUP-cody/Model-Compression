# 工作日志（精简）

> 详细版（结论与创新点）：[WORK_LOG.md](WORK_LOG.md)  
> 日期：2026-08-13（压缩比验证：2026-08-14）

## 已做

1. **MNIST P1.2 GPU 统计补全**：修好 dense_small 从零训练；跑完 3 seed × 6 压缩率；迭代/自主搜索在高压缩下仍约 97–98% test acc，one-shot 超过 2x 后崩溃。
2. **P2 代码**：CIFAR train/val/test 协议、ResNet-18、块内 conv1 通道剪枝、CNN Wanda、六方法对照 runner、LoRA/自蒸馏接口。已推送 `b60d4e9`。
3. **CIFAR 基线 + smoke（已跑通）**  
   - 20 epoch 基线 val **88.86%**，test **87.88%**  
   - smoke 目录：`results/cifar_p12_comparison_gpu_smoke/cifar_p12_comparison_cifar_p12_gpu_smoke_a145fe93fd29_1`  
   - 轻剪 + 无恢复：oneshot/搜索 test 约 86–87%  
   - `dense_small` ≈ 10%：smoke 未训练小模型，不是算法失败  
   - 修了剪枝 checkpoint 无法 report-test、GPU benchmark 参数冲突
4. **CIFAR formal + 3-seed multiseed（已跑通）**  
   - Formal：`results/cifar_p12_comparison_gpu/cifar_p12_comparison_cifar_p12_gpu_study_12815300d99f`（study 134 s + report-test 20 s）  
   - Multiseed：`results/cifar_p12_comparison_gpu_multiseed/`（3 seed，约 229 s）  
   - dense_small 3 epoch 训练后 test 约 **67%**（不再是 10%）  
   - 当时 YAML 只剪 1–2 个 conv1，实际压缩约 **1.00x**，不是 GPU 慢
5. **CNN 目标压缩比（已验证）**  
   - conv1-only 上限：均匀 0.95 约 **15.1x**，2x/4x/10x 都够，未剪残差输出  
   - 配置改为 `target_compression_ratio` 二分推导全部 8 个 conv1  
   - 2x smoke：`cifar_p12_gpu_smoke_866113f8e78a`，剪枝臂实际压缩 **2.01x**（门禁 1.7–2.3）  
   - 无恢复时 oneshot/search test 掉到约 23–46%，只证明压缩到位，不能比方法

## 现在处于哪一步

2x 压缩已打到。全量 sweep 和带恢复的 formal 还没跑。

## 下一步

CIFAR 压缩率扫描（6×3）→ 带 recovery 的 2x formal → 恢复消融。
